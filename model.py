import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,1]])  # .pow(.5)
        an_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,2]]) # .pow(.5)
        losses = F.relu(an_cos - ap_cos + self.margin)
        return losses.mean(), len(triplets)

class Encoder(nn.Module):
    def __init__(self, expansion, blocks, embedding_dim):
        super(Encoder, self).__init__()
        self.convnet = self._make_layers(expansion, blocks)
        self.pool = nn.AdaptiveAvgPool2d([2,2])
        self.fc = nn.Linear(expansion * blocks * 2 * 2 * 2, embedding_dim)
    
    def _make_layers(self, expansion, blocks):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(nn.Conv2d(1, expansion, kernel_size = 5, stride = 2, padding = 2))
            else:
                layers.append(nn.Conv2d(expansion, expansion * 2, kernel_size = 5, stride = 2, padding = 2))
                expansion = expansion * 2
            layers.append(nn.BatchNorm2d(expansion))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convnet(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return F.normalize(x)

class Generator(nn.Module):
    def __init__(self, expansion, blocks, embedding_dim, input_dim, latent_dim = 32):
        super(Generator, self).__init__()
        dim = input_dim // (2 ** blocks) # 8
        channel = expansion * (2 ** (blocks - 1)) # 256
        self.dim = dim
        self.channel = channel
        self.fc = nn.Linear(embedding_dim + latent_dim, self.channel * self.dim ** 2)
        self.bn = nn.BatchNorm1d(self.channel * self.dim ** 2)
        # self.convTranspose = self._make_layers(channel, blocks)
        self.convTranspose = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size = 5, stride = 3, padding = 5),
                                           nn.BatchNorm2d(128),
                                           nn.LeakyReLU(0.2),
                                           nn.ConvTranspose2d(128,64,kernel_size = 5, stride = 3, padding = 9),
                                           nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.2),
                                           nn.ConvTranspose2d(64,32,kernel_size = 5, stride = 3, padding = 17),
                                           nn.BatchNorm2d(32),
                                           nn.LeakyReLU(0.2),
                                           nn.ConvTranspose2d(32,1,kernel_size = 5, stride = 3, padding = 33),
                                          )
        self.tanh = nn.Tanh()

    def _make_layers(self, channel, blocks):
        layers = []
        for i in range(blocks):
            if i == blocks - 1:
                layers.append(nn.ConvTranspose2d(channel * 2, 1, kernel_size = 5, stride = 3, padding = (1 + 4 * (i + 1))))
            else:
                layers.append(nn.ConvTranspose2d(channel, channel // 2, kernel_size = 5, stride = 3, padding = (1 + 4 * (i + 1))))
                layers.append(nn.BatchNorm2d(channel // 2))
                layers.append(nn.LeakyReLU(0.2))
                channel = channel // 2
        return nn.Sequential(*layers)

    def forward(self, embedding, x):
        x = torch.cat((embedding, x), -1)
        print(x.size())
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel, self.dim, self.dim)
        x = self.convTranspose(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, expansion, blocks, embedding_dim):
        super(Discriminator, self).__init__()
        self.convnet = self._make_layers(expansion, blocks)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layers(self, expansion, blocks):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(nn.Conv2d(1, expansion, kernel_size = 5, stride = 2, padding = 2))
            else:
                layers.append(nn.Conv2d(expansion, expansion * 2, kernel_size = 5, stride = 2, padding = 2))
                expansion = expansion * 2
            layers.append(nn.BatchNorm2d(expansion))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Classifier(nn.Module):
    def __init__(self, expansion, blocks, n_classes):
        super(Classifier, self).__init__()
        self.convnet = self._make_layers(expansion, blocks)
        self.fc1 = nn.Linear(256 * 8 * 8, 2 * n_classes)
        self.bn1 = nn.BatchNorm1d(2 * n_classes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * n_classes, n_classes)
    
    def _make_layers(self, expansion, blocks):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(nn.Conv2d(1, expansion, kernel_size = 5, stride = 2, padding = 2))
            else:
                layers.append(nn.Conv2d(expansion, expansion * 2, kernel_size = 5, stride = 2, padding = 2))
                expansion = expansion * 2
            layers.append(nn.BatchNorm2d(expansion))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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
    def __init__(self, expansion, blocks, embedding_dim, input_dim, latent_dim = 512):
        super(Generator, self).__init__()
        dim = input_dim // (2 ** blocks) # 8
        channel = expansion * (2 ** (blocks - 1)) # 256
        self.dim = dim
        self.channel = channel
        self.fc = nn.Linear(embedding_dim + latent_dim, self.channel * self.dim ** 2)
        self.bn = nn.BatchNorm1d(self.channel * self.dim ** 2)
        self.upsample = self._make_layers(channel, blocks)
        self.up = nn.Upsample(scale_factor = 2)
        self.conv = nn.Conv2d(expansion, 1, kernel_size = 3, stride = 1, padding = 1)
        self.tanh = nn.Tanh()

    def _make_layers(self, channel, blocks):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1))
                layers.append(nn.BatchNorm2d(channel))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Upsample(scale_factor = 2))
                layers.append(nn.Conv2d(channel, channel // 2, kernel_size = 3, stride = 1, padding = 1))
                layers.append(nn.BatchNorm2d(channel // 2))
                layers.append(nn.LeakyReLU(0.2))
                channel = channel // 2
        return nn.Sequential(*layers)

    def forward(self, embedding, x):
        x = torch.cat((embedding, x), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel, self.dim, self.dim)
        x = self.upsample(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, expansion, blocks, embedding_dim):
        super(Discriminator, self).__init__()
        self.convnet = self._make_layers(expansion, blocks)
        self.pool = nn.AdaptiveAvgPool2d([2,2])
        self.fc1 = nn.Linear(expansion * blocks * 2 * 2 * 2, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, 1)
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
        x = self.pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Classifier(nn.Module):
    def __init__(self, expansion, blocks, embedding_dim, n_classes):
        super(Classifier, self).__init__()
        self.convnet = self._make_layers(expansion, blocks)
        self.pool = nn.AdaptiveAvgPool2d([2,2])
        self.fc1 = nn.Linear(expansion * blocks * 2 * 2 * 2, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, n_classes)
    
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
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

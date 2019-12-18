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

    def __init__(self, margin, triplet_selector, device):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.device = device

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.to(self.device)
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

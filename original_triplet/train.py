import warnings
warnings.filterwarnings('ignore')
import torch
from torchvision.utils import save_image
import os
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from dataset import VAD, TRIAL_FILE, FEATURE_LEN
from dataset import SpeakerTrainDataset, SpeakerTestDataset, BalancedBatchSampler
from naive_model import Encoder, OnlineTripletLoss
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from utils import RandomNegativeTripletSelector
from utils import eer as cal_eer
import yaml
from logger import Logger
import time
from eval_metric import AverageNonzeroTripletsMetric

logger = Logger('log/{}'.format(time.ctime()).replace(' ', '_'))

f = open('./config/config.yaml', 'r')
config = yaml.load(f)
f.close()
model_config = config['model']
train_config = config['train']

model_dir = train_config['model_dir']
final_dir = train_config['final_dir']
epochs = train_config['epochs']
optimizer = train_config['optimizer']
momentum = train_config['momentum']
lr = train_config['lr']
lr_decay = train_config['lr_decay']
weight_decay = train_config['weight_decay']
resume = train_config['resume']
load_optimizer = train_config['load_optimizer']
seed = train_config['seed']
prediction = train_config['prediction']
spks_per_batch = train_config['spks_per_batch']
utts_per_spk = train_config['utts_per_spk']
# start = train_config['start']

embedding_dim = model_config['embedding_dim']
margin = model_config['margin']
blocks = model_config['blocks']
expansion = model_config['expansion']

device = torch.device('cuda:1')
os.makedirs(model_dir, exist_ok = True)
os.makedirs(final_dir, exist_ok = True)
# os.makedirs('fake_fbank', exist_ok = True)

def train(epoch, 
          encoder,
          triplet_criterion, 
          encoder_optimizer, 
          train_loader,
          metric): # 训练轮数，模型，loss，优化器，数据集读取
    encoder.train() # 初始化模型为训练模式

    adjust_learning_rate(encoder_optimizer, epoch) # 调整学习率

    triplet_sum_loss, sum_samples = 0, 0
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in progress_bar:
        sum_samples += len(data)
        data = data.to(device)
        label = label.to(device)  # 数据和标签

        # triplet loss train
        embedding = encoder(data)
        triplet_loss, non_zero_triplets = triplet_criterion(embedding, label) # loss
        metric(non_zero_triplets)
        logger.log_value('non_zero_triplets', metric.value())
        triplet_loss = triplet_loss
        encoder_optimizer.zero_grad()
        triplet_loss.backward() # bp训练
        encoder_optimizer.step()

        triplet_sum_loss += triplet_loss.item() * len(data)
        logger.log_value('triplet_loss', triplet_sum_loss / sum_samples)

        logger.step()

        progress_bar.set_description(
            'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] TriLoss: {:.4f} Nonzero: {:.4f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                triplet_sum_loss / sum_samples,
                metric.value()))


    torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict()},
                '{}/net_{}.pth'.format(model_dir, epoch)) # 保存当轮的模型到net_{}.pth
    torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict()},
                '{}/net.pth'.format(final_dir)) # 保存当轮的模型到net.pth

def test(model, test_loader): # 测试，模型，测试集读取
    model.eval() # 设置为测试模式

    pairs, similarities_embedding = [], []
    progress_bar = tqdm(enumerate(test_loader))
    for batch_idx, (pair, data1, data2) in progress_bar: # 按batch读取数据
        pairs.append(pair)
        with torch.no_grad():
            data1, data2 = data1.to(device), data2.to(device)

            embedding1 = model(data1)
            embedding2 = model(data2)
            sim_embedding = F.cosine_similarity(embedding1, embedding2).cpu().data.numpy()
            similarities_embedding.append(sim_embedding)

            progress_bar.set_description('Test: [{}/{} ({:3.3f}%)]'.format(
                batch_idx + 1, len(test_loader), 100. * (batch_idx + 1) / len(test_loader)))

    pairs = np.concatenate(pairs)

    similarities_embedding = np.array([sub_sim for sim in similarities_embedding for sub_sim in sim])

    with open(final_dir + prediction, mode='w') as f:
        f.write('pairID,pred\n')
        for i in range(len(similarities_embedding)):
            f.write('{},{}\n'.format(pairs[i], similarities_embedding[i]))

def main():
    torch.manual_seed(seed) # 设置随机种子

    train_dataset = SpeakerTrainDataset() # 设置训练集读取
    n_classes = train_dataset.n_classes # 说话人数
    batch_sampler = BalancedBatchSampler(train_dataset.labels, 
                                         train_dataset.count, 
                                         spks_per_batch,
                                         utts_per_spk)
    print('Num of classes: {}'.format(n_classes))

    encoder = Encoder(expansion, blocks, embedding_dim).to(device)
    # encoder = Encoder().to(device)

    if optimizer == 'sgd': # 优化器使用sgd
        encoder_optimizer = optim.SGD(encoder.parameters(), lr = lr, momentum = momentum, dampening = 0, weight_decay = weight_decay)
    elif optimizer == 'adagrad': # 优化器使用adagrad
        encoder_optimizer = optim.Adagrad(encoder.parameters(), lr = lr, lr_decay = lr_decay,  weight_decay = weight_decay)
    else: # 优化器使用adam
        encoder_optimizer = optim.Adam(encoder.parameters(), lr = lr, weight_decay = weight_decay)
    
    selector = RandomNegativeTripletSelector(margin)

    triplet_criterion = OnlineTripletLoss(margin, selector, device)

    start = 1
    metric = AverageNonzeroTripletsMetric()

    if resume: # 是否从之前保存的模型开始
        if os.path.isfile(os.path.join(final_dir, "net.pth")):
            print('=> loading checkpoint {}'.format(os.path.join(final_dir, "net.pth")))
            checkpoint = torch.load(os.path.join(final_dir, "net.pth"))
            start = checkpoint['epoch'] + 1
            if load_optimizer:
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(os.path.join(final_dir, "net.pth")))

    train_loader = DataLoader(train_dataset, 
                              batch_sampler = batch_sampler,
                              num_workers=8, 
                              pin_memory=True)

    test_dataset = SpeakerTestDataset()
    test_loader = DataLoader(test_dataset, 
                             batch_size = 1, 
                             shuffle=False,
                             num_workers=8, 
                             pin_memory=True)

    for epoch in range(start, epochs + 1):
        fake_fbank = train(epoch, 
                           encoder, 
                           triplet_criterion, 
                           encoder_optimizer, 
                           train_loader,
                           metric)
        test(encoder, test_loader) # 测试
        task = pd.read_csv(TRIAL_FILE, header=None, delimiter = '[ ]', engine='python')
        pred = pd.read_csv(final_dir + prediction, delimiter = '[,]', engine='python')
        y_true = task.iloc[:, 0].values
        y_pred = pred.iloc[:, -1].values
        eer, thresh = cal_eer(y_true, y_pred)
        print('\nEER      : {:.3%}'.format(eer))
        print('Threshold: {:.5f}'.format(thresh))
        logger.log_value('eer', eer)

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 15:
        adjusted_lr = lr
    elif epoch <= 30:
        adjusted_lr = lr * 0.1
    else:
        adjusted_lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = adjusted_lr
       
main()

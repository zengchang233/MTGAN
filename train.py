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
from model import Encoder, Generator, Discriminator, Classifier, OnlineTripletLoss
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
from torch.optim import lr_scheduler
import random

logger = Logger('log/{}'.format(time.ctime()).replace(' ', '_'))

f = open('config/config.yaml', 'r')
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

random.seed(seed)

embedding_dim = model_config['embedding_dim']
margin = model_config['margin']
blocks = model_config['blocks']
expansion = model_config['expansion']
latent_dim = model_config['latent_dim']

device = torch.device('cuda')
os.makedirs(model_dir, exist_ok = True)
os.makedirs(final_dir, exist_ok = True)
os.makedirs('fake_fbank', exist_ok = True)

def train(epoch, 
          encoder,
          generator,
          discriminator,
          classifier,
          triplet_criterion, 
          bce_criterion,
          softmax_criterion,
          generator_optimizer,
          discriminator_optimizer,
          train_loader,
          metric): # 训练轮数，模型，loss，优化器，数据集读取
    encoder.train() # 初始化模型为训练模式
    generator.train()
    discriminator.train()
    classifier.train()

    triplet_sum_loss, generator_loss, discriminator_loss, classifier_loss, sum_samples = 0, 0, 0, 0, 0
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in progress_bar:
        sum_samples += len(data)
        data = data.to(device)
        label = label.to(device)  # 数据和标签

        # Adversarial ground truths
        valid = torch.FloatTensor(utts_per_spk * spks_per_batch, 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(utts_per_spk * spks_per_batch, 1).fill_(0.0).to(device)

        # triplet loss
        generator_optimizer.zero_grad()
        embedding = encoder(data)
        triplet_loss, non_zero_triplets = triplet_criterion(embedding, label) # loss
        metric(non_zero_triplets)
        triplet_sum_loss += triplet_loss.item() * len(data)
        logger.log_value('non_zero_triplets', metric.value())
        logger.log_value('triplet_loss', triplet_sum_loss / sum_samples)

        # generator loss
        z = torch.rand(spks_per_batch * utts_per_spk, latent_dim).to(device) # 生成噪声
        fake_fbank = generator(embedding, z) # 生成fake fbank
        validity = discriminator(fake_fbank) # 判别器判断
        g_loss = bce_criterion(validity, valid) # 计算loss
        logger.log_value('g_loss', g_loss)
        triplet_g_loss = 0.2 * g_loss + 0.1 * triplet_loss
        triplet_g_loss.backward() # bp训练
        generator_optimizer.step()

        generator_loss += g_loss.item() * len(data)

        # discriminator loss train
        discriminator_optimizer.zero_grad()
        validity_real = discriminator(data) 
        d_real_loss = bce_criterion(validity_real, valid) # Loss for real images
        validity_fake = discriminator(fake_fbank.detach()) 
        d_fake_loss = bce_criterion(validity_fake, fake) # Loss for fake images
        d_loss = (d_real_loss + d_fake_loss) / 2 # Total discriminator loss
        logger.log_value('d_loss', d_loss)
        discriminator_loss += d_loss.item() * len(data)
        
        # softmax loss train
        input = torch.cat((data, fake_fbank.detach()), 0)
        label = torch.cat((label, label), -1)
        output = classifier(input)
        softmax_loss = softmax_criterion(output, label)
        logger.log_value('soft_loss', softmax_loss)
        classifier_loss += softmax_loss.item() * len(data)

        discriminator_classifier_loss = 0.2 * softmax_loss + 0.5 * d_loss
        discriminator_classifier_loss.backward()
        discriminator_optimizer.step()

        logger.step()

        progress_bar.set_description(
            'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] TriLoss: {:.4f} Nonzero: {:.4f} GenLoss: {:.4f} DisLoss: {:.4f} ClaLoss: {:.4f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                triplet_sum_loss / sum_samples,
                metric.value(),
                generator_loss / sum_samples,
                discriminator_loss / sum_samples,
                classifier_loss / sum_samples))

    torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()},
                '{}/net_{}.pth'.format(model_dir, epoch)) # 保存当轮的模型到net_{}.pth
    torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()},
                '{}/net.pth'.format(final_dir)) # 保存当轮的模型到net.pth
    return data.detach(), fake_fbank.detach()

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
    generator = Generator(expansion, blocks, embedding_dim, FEATURE_LEN, latent_dim).to(device)
    discriminator = Discriminator(expansion, blocks, embedding_dim).to(device)
    classifier = Classifier(expansion, blocks, n_classes).to(device)

    if optimizer == 'sgd': # 优化器使用sgd
        generator_optimizer = optim.SGD([{'params': generator.parameters()}, 
                                         {'params': encoder.parameters()},
                                         {'params': discriminator.parameters()}],
                                        lr = lr, momentum = momentum, weight_decay = weight_decay)
        discriminator_optimizer = optim.SGD([{'params': discriminator.parameters()}, 
                                             {'params': classifier.parameters()}],
                                            lr = lr, momentum = momentum, weight_decay = weight_decay)
    elif optimizer == 'adagrad': # 优化器使用adagrad
        generator_optimizer = optim.Adagrad([{'params': generator.parameters()}, 
                                             {'params': encoder.parameters()},
                                             {'params': discriminator.parameters()}], 
                                            lr = lr, weight_decay = weight_decay)
        discriminator_optimizer = optim.Adagrad([{'params': discriminator.parameters()}, 
                                                 {'params': classifier.parameters()}], 
                                                lr = lr, weight_decay = weight_decay)
    else: # 优化器使用adam
        generator_optimizer = optim.Adam([{'params': generator.parameters()}, 
                                         {'params': encoder.parameters()},
                                         {'params': discriminator.parameters()}], 
                                        lr = lr, weight_decay = weight_decay)
        discriminator_optimizer = optim.Adam([{'params': discriminator.parameters()}, 
                                              {'params': classifier.parameters()}], 
                                             lr = lr, weight_decay = weight_decay)
    
    gen_scheduler = lr_scheduler.StepLR(generator_optimizer, 200, gamma = 0.1, last_epoch = -1)
    dis_scheduler = lr_scheduler.StepLR(discriminator_optimizer, 200, gamma = 0.1, last_epoch = -1)
    selector = RandomNegativeTripletSelector(margin)

    triplet_criterion = OnlineTripletLoss(margin, selector).to(device)
    bce_criterion = nn.BCELoss().to(device)
    softmax_criterion = nn.CrossEntropyLoss().to(device)

    start = 1
    metric = AverageNonzeroTripletsMetric()

    if resume: # 是否从之前保存的模型开始
        if os.path.isfile(os.path.join(final_dir, "net.pth")):
            print('=> loading checkpoint {}'.format(os.path.join(final_dir, "net.pth")))
            checkpoint = torch.load(os.path.join(final_dir, "net.pth"))
            start = checkpoint['epoch'] + 1
            if load_optimizer:
                generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
                discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
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
        real_data, fake_fbank = train(epoch, 
                                      encoder, 
                                      generator,
                                      discriminator,
                                      classifier,
                                      triplet_criterion, 
                                      bce_criterion,
                                      softmax_criterion,
                                      generator_optimizer,
                                      discriminator_optimizer,
                                      train_loader,
                                      metric)
        save_image(torch.Tensor(random.sample(real_data.tolist(), 9)),
                   'real_fbank/{}.png'.format(epoch),
                    nrow = 2, normalize = True)                                      
        save_image(torch.Tensor(random.sample(fake_fbank.tolist(), 9)), 
                   'fake_fbank/{}.png'.format(epoch), 
                   nrow = 2, normalize = True)
        test(encoder, test_loader) # 测试
        gen_scheduler.step()
        dis_scheduler.step()
        task = pd.read_csv(TRIAL_FILE, header=None, delimiter = '[,]', engine='python')
        pred = pd.read_csv(final_dir + prediction, engine='python')
        y_true = np.array(task.iloc[:, 0])
        y_pred = np.array(pred.iloc[:, -1])
        eer, thresh = cal_eer(y_true, y_pred)
        logger.log_value('eer', eer, epoch)
        print('EER      : {:.3%}'.format(eer))
        print('Threshold: {:.5f}'.format(thresh))
       
main()
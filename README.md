# MTGAN
[MTGAN: Speaker Verification through Multitasking Triplet Generative Adversarial Networks](https://arxiv.org/abs/1803.09059)

#### **Abstract** 

In this paper, we propose an enhanced triplet method that improves the encoding process of embeddings by jointly utilizing generative adversarial mechanism and multitasking optimization. We extend our triplet encoder with Generative Adversarial Networks (GANs) and softmax loss function. GAN is introduced for increasing the generality and diversity of samples, while softmax is for reinforcing features about speakers. For simplification, we term our method Multitasking Triplet Generative Adversarial Networks (MTGAN). Experiment on short utterances demonstrates that MTGAN reduces the verification equal error rate (EER) by 67% (relatively) and 32% (relatively) over conventional i-vector method and state-of-the-art triplet loss method respectively. This effectively indicates that MTGAN outperforms triplet methods in the aspect of expressing the high-level feature of speaker information.

## Instruction

This is an **unofficial MTGAN implementation**. It only provides a preliminary code for these neural network of this model architecture **without** concretely calculation such as convolutional kernel size and pooling kernel size.

In this repository, I used cosine similarity-based Tripelt loss instead of the Euclidean distance-based Triplet loss used by the authors of this paper.

## Dataset

Private dataset. Anybody using this code can download Voxceleb dataset as substitution.

## Original Triplet Loss

In original_triplet directory, I implemented a simple triplet loss which uses a randomly hard sampling strategy. It is treated as a contrastive experiment to MTGAN so the encoder of it is same with MTGAN.

After 45 epochs, the model is converged. Changing trend of loss function is shown as the following picture.

Triplet loss

![triplet loss](https://github.com/zengchang94622/MTGAN/blob/master/imgs/triplet_loss.png)

Number of non zero triplets

![non zero triplets](https://github.com/zengchang94622/MTGAN/blob/master/imgs/non_zero_triplets.png)

EER

![eer](https://github.com/zengchang94622/MTGAN/blob/master/imgs/eer.png)

EER: 7.643%

## I-Vector System

I used KALDI sre16 scripts to build an I-Vector system.

EER: 3.422%

## ResNet18 + LMCL

Another neural network-based system was built with augmenting data by using MUSAN.

EER: 2.23%

## MTGAN Performance

I have developed a preliminary version yet without adjust hyper-parameters carefully, so the performance of this system is not well as original triplet loss.

EPOCH:     38

EER:       11.701%

## Contact

Email:  zengchang.elec@gmail.com

WeChat: zengchang-_-

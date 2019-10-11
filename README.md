# MTGAN
[MTGAN: Speaker Verification through Multitasking Triplet Generative Adversarial Networks](https://arxiv.org/abs/1803.09059)

\[**Abstract**\]In this paper, we propose an enhanced triplet method that improves the encoding process of embeddings by jointly utilizing
generative adversarial mechanism and multitasking optimization. We extend our triplet encoder with Generative Adversarial Networks (GANs) and softmax loss function. GAN is introduced for increasing the generality and diversity of samples, while softmax is for reinforcing features about speakers. For simplification, we term our method Multitasking Triplet Generative Adversarial Networks (MTGAN). Experiment on short utterances demonstrates that MTGAN reduces the verification equal error rate (EER) by 67% (relatively) and 32% (relatively) over conventional i-vector method and state-of-the-art triplet loss method respectively. This effectively indicates that MTGAN outperforms triplet methods in the aspect of expressing the high-level feature of speaker information. Index Terms: generative adversarial networks, speaker verification, triplet loss

## Instruction

This is an **unofficial MTGAN implementation**. It only provides a preliminary code for these neural network of this model architecture **without** concretely calculation such as convolutional kernel size and pooling kernel size.

In this repository, I used cosine similarity-based Tripelt loss instead of the Euclidean distance-based Triplet loss used by the authors of this paper.

## Dataset

Private dataset. Anybody using this code can download Voxceleb dataset as substitution.

## Performance

EPOCH:     38
EER:       11.701%
Threshold: 0.42206

## Contact

Email:  zengchang.elec@gmail.com
WeChat: zengchang-_-

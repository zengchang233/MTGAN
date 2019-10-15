import warnings
warnings.filterwarnings('ignore')
import csv
import math
import os
import random
import librosa
import numpy as np
import soundfile as sf
import torch
from python_speech_features import mfcc, fbank, delta
from scipy.signal.windows import hamming
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from torch.utils.data import BatchSampler
import yaml
import pandas as pd

f = open('config.yaml', 'r')
config = yaml.load(f)['data']
f.close()

SAMPLE_RATE = config['sample_rate']
VAD = config['vad']
FEATURE = config['feature']
FEATURE_LEN = config['feature_len']
WIN_LEN = config['win_len']
WIN_STEP = config['win_step']
N_FRAMES = config['n_frames']
TRAIN_MANIFEST = config['manifest']
TRIAL_FILE = config['trial_file']
ENROLL_WAV = config['enroll']
VERIFY_WAV = config['verify']
DATA_PATH = config['data_path']
TEST_MODE = config['test_mode']

N_FFT = int(WIN_LEN * SAMPLE_RATE)
HOP_LEN = int(WIN_STEP * SAMPLE_RATE)
DURATION = (N_FRAMES - 1) * WIN_STEP + WIN_LEN # 固定为300帧，300帧窗口移动299次
N_SAMPLES = int(DURATION * SAMPLE_RATE)

ENROLL_FEATURE = '/root/datasets/all_labelled_voice/{}_features/{}/enroll'.format(TEST_MODE, FEATURE)
VERIFY_FEATURE = '/root/datasets/all_labelled_voice/{}_features/{}/verify'.format(TEST_MODE, FEATURE)

def load_audio(filename, start = 0, stop = None, resample = True):
    y = None
    sr = SAMPLE_RATE
    y, sr = sf.read(filename, start = start, stop = stop, dtype = 'float32', always_2d = True)
    y = y[:, 0]
    return y, sr

def normalize(v):
    return (v - v.mean(axis = 0)) / (v.std(axis = 0) + 2e-12)

def make_feature(y, sr):
    if FEATURE == 'fft':
        S = librosa.stft(y, n_fft = N_FFT, hop_length = HOP_LEN, window = hamming)          
        feature, _ = librosa.magphase(S)
        feature = np.log1p(feature)
        feature = feature.transpose()
    else:
        if FEATURE == 'fbank':
            feature, _ = fbank(y, sr, winlen = WIN_LEN, winstep = WIN_STEP, nfilt = FEATURE_LEN)
        else:
            feature = mfcc(y, sr, winlen = WIN_LEN, winstep = WIN_STEP)
            feature_d1 = delta(feature, N = 1)
            feature_d2 = delta(feature, N = 2)
            feature = np.hstack([feature, feature_d1, feature_d2])
    return normalize(feature).astype(np.float32)

def process_avg_test_dataset():
    test = pd.read_csv(TRIAL_FILE, header = None, engine = 'python')
    enroll_speakers = test.iloc[:,1]
    test_wav = test.iloc[:,2]
    for speaker in tqdm(list(set(enroll_speakers))):
        speaker_path = os.path.join(ENROLL_WAV, str(speaker))
        os.makedirs(os.path.join(ENROLL_FEATURE, str(speaker)), exist_ok = True)
        for speech in os.listdir(speaker_path):
            if speech[0] != '.':
                speech_path = os.path.join(speaker_path, speech)
                y, sr = load_audio(speech_path)
                feature = make_feature(y, sr)
                feature_path = os.path.join(ENROLL_FEATURE, str(speaker), speech.replace('.wav', '.npy'))
                np.save(feature_path, feature)
    for speech in tqdm(test_wav):
        speech_path = os.path.join(VERIFY_WAV, speech)
        feature_path = os.path.join(VERIFY_FEATURE, speech.replace('.wav', '.npy'))
        speaker_path = os.path.join(VERIFY_FEATURE, speech.split('/')[0])
        os.makedirs(os.path.join(speaker_path), exist_ok = True)
        if not os.path.exists(feature_path):
            y, sr = load_audio(speech_path)
            feature = make_feature(y, sr)
            np.save(feature_path, feature)        
    print('done')

def process_concat_test_dataset():
    random.seed(10)
    test = pd.read_csv(TRIAL_FILE, header = None, engine = 'python')
    enroll_speakers = test.iloc[:,1]
    test_wav = test.iloc[:,2]
    for speaker in tqdm(list(set(enroll_speakers))):
        speaker_path = os.path.join(ENROLL_WAV, str(speaker))
        os.makedirs(os.path.join(ENROLL_FEATURE, str(speaker)), exist_ok = True)
        enroll_feature = []
        for speech in os.listdir(speaker_path):
            if speech[0] != '.':
                speech_path = os.path.join(speaker_path, speech)
                y, sr = load_audio(speech_path)
                feature = make_feature(y, sr)
                enroll_feature.append(feature)
        feature_path = os.path.join(ENROLL_FEATURE, str(speaker), 'speaker_{}'.format(speaker) + '.npy')
        np.save(feature_path, np.vstack(enroll_feature))
    for speech in tqdm(test_wav):
        speech_path = os.path.join(VERIFY_WAV, speech)
        feature_path = os.path.join(VERIFY_FEATURE, speech.replace('.wav', '.npy'))
        speaker_path = os.path.join(VERIFY_FEATURE, speech.split('/')[0])
        os.makedirs(os.path.join(speaker_path), exist_ok = True)
        if not os.path.exists(feature_path):
            y, sr = load_audio(speech_path)
            feature = make_feature(y, sr)
            np.save(feature_path, feature)        
    print('done')

'''
os.makedirs(ENROLL_FEATURE, exist_ok = True)
os.makedirs(VERIFY_FEATURE, exist_ok = True)

if TEST_MODE == 'concat':
    print('processing concat feature')
    process_concat_test_dataset()
if TEST_MODE == 'avg':
    print('processing avg feature')
    process_avg_test_dataset()
'''

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, all_speech, n_classes, n_samples):
        self.labels = list(set(labels))
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = all_speech
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices += [class_ for i in range(self.n_samples)]
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class SpeakerTrainDataset(Dataset):
    def __init__(self):
        '''
        dataset，保存每个人的语音，每个人的所有语音放在一个数组里面，每条语音的信息
        放在一个元组里面。所有人的语音放在dataset里面
        '''
        self.dataset = []
        current_sid = -1
        self.count = 0
        self.labels = []
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, _, filename, duration, samplerate in reader:
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))  
                self.count += 1  
                self.labels.append(eval(sid))
        self.n_classes = len(self.dataset)

    def __len__(self):
        return self.count
    
    def __getitem__(self, sid):
        sid %= self.n_classes #数据集长度可能大于说话人长度，每个说话人取多少个片段也很关键
        speaker = self.dataset[sid]
        y = []
        n_samples = 0
        while n_samples < N_SAMPLES:
            aid = random.randrange(0, len(speaker)) # 从当前sid的里面随机选择一条语音
            audio = speaker[aid]
            t, sr = audio[1], audio[2] # duration和sample rate
            if t < 1.0: # 如果少于1秒，跳过不看
                continue
            if n_samples == 0:
                start = int(random.uniform(0, t - 1.0) * sr) # 找到截断的开头
            else:
                start = 0
            stop = int(min(t, max(1.0, (start + N_SAMPLES - n_samples) / SAMPLE_RATE)) * sr)
            _y, _ = load_audio(audio[0], start = start, stop = stop)
            if _y is not None:
                y.append(_y)
                n_samples += len(_y)
        # 返回特征和说话人id
        return np.array([make_feature(np.hstack(y)[:N_SAMPLES], SAMPLE_RATE).transpose()]), sid

class SpeakerTestDataset(Dataset):
    def __init__(self):
        self.features = []
        self.pairID = []
        with open(TRIAL_FILE) as f:
            pairs = f.readlines()
            for pair in pairs:
                pair = pair[2:]
                pair_list = pair.split(',')
                self.pairID.append(pair.strip())
                self.features.append((os.path.join(ENROLL_FEATURE, pair_list[0], 'speaker_{}.npy'.format(pair_list[0])),
                                      os.path.join(VERIFY_FEATURE, '{}.npy'.format(pair_list[1].split('.')[0]))))

    def __getitem__(self, index):
        return self.pairID[index], np.array([np.load(self.features[index][0]).transpose()]),\
                np.array([np.load(self.features[index][1]).transpose()])

    def __len__(self):
        return len(self.features) 

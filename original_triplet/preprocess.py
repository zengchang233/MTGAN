import csv
import os

import soundfile as sf
from tqdm import tqdm
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys
import numpy as np
import config
import warnings
import yaml
warnings.filterwarnings('ignore')

f = open('./config/config.yaml', 'r')
config = yaml.load(f)['data']
f.close()
data_path = config['data_path']

SAMPLE_RATE = 16000
os.makedirs('../manifest', exist_ok = True)
MANIFEST_DIR = '../manifest/{}_manifest.csv'

def read_manifest(dataset, start = 0):
    n_speakers = 0
    rows = []
    with open(MANIFEST_DIR.format(dataset), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            rows.append([int(sid) + start, aid, filename, duration, samplerate])
            n_speakers = int(sid) + 1
    return n_speakers, rows

def save_manifest(dataset, rows):
    rows.sort()
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def create_manifest_voxceleb1():
    dataset = 'voxceleb1'
    n_speakers = 0
    log = []
    train_dataset = os.path.join(data_path, 'wav')
    for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
        speaker_dir = os.path.join(train_dataset, speaker)
        aid = 0
        for sub_speaker in os.listdir(speaker_dir):
            sub_speaker_path = os.path.join(speaker_dir, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1) and (audio.find('-') == -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                        aid += 1
        n_speakers += 1
    save_manifest(dataset, log)

def cal_eer(y_true, y_pred):
    fpr, tpr, thresholds= roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

if __name__ == '__main__':
    create_manifest_voxceleb1()

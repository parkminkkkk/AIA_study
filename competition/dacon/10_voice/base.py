import random
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import librosa
from autogluon.tabular import TabularDataset, TabularPredictor

import warnings
warnings.filterwarnings(action='ignore')

# path = './_data/_dacon_emotion_recognition/train/'
# save_path = './_save/dacon_emotion_recognition/test/'

CFG = {
    'SR':16000,
    'N_MFCC':32, # Melspectrogram 벡터를 추출할 개수
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정

train_df = pd.read_csv('d:/_study/_data/voice/train.csv')
test_df = pd.read_csv('d:/_study/_data/voice/test.csv')

train_df['path'] = 'd:/study/_data/voice/' + train_df['path']
test_df['path'] = 'd:/study/_data/voice/' + test_df['path']

def get_mfcc_feature(df):
    features = []
    for path in tqdm(df['path']):
        y, sr = librosa.load(path, sr=CFG['SR'])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        features.append({
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_max': np.max(mfcc, axis=1),
            'mfcc_min': np.min(mfcc, axis=1),
        })

    mfcc_df = pd.DataFrame(features)
    mfcc_mean_df = pd.DataFrame(mfcc_df['mfcc_mean'].tolist(), columns=[f'mfcc_mean_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_max_df = pd.DataFrame(mfcc_df['mfcc_max'].tolist(), columns=[f'mfcc_max_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_min_df = pd.DataFrame(mfcc_df['mfcc_min'].tolist(), columns=[f'mfcc_min_{i}' for i in range(CFG['N_MFCC'])])

    return pd.concat([mfcc_mean_df, mfcc_max_df, mfcc_min_df], axis=1)


def get_feature_mel(df):
    features = []
    for path in tqdm(df['path']):
        data, sr = librosa.load(path, sr=CFG['SR'])
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 128

        D = np.abs(librosa.stft(data, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

        features.append({
            'mel_mean': mel.mean(axis=1),
            'mel_max': mel.min(axis=1),
            'mel_min': mel.max(axis=1),
        })
    mel_df = pd.DataFrame(features)
    mel_mean_df = pd.DataFrame(mel_df['mel_mean'].tolist(), columns=[f'mel_mean_{i}' for i in range(n_mels)])
    mel_max_df = pd.DataFrame(mel_df['mel_max'].tolist(), columns=[f'mel_max_{i}' for i in range(n_mels)])
    mel_min_df = pd.DataFrame(mel_df['mel_min'].tolist(), columns=[f'mel_min_{i}' for i in range(n_mels)])

    return pd.concat([mel_mean_df, mel_max_df, mel_min_df], axis=1)

train_mf = get_mfcc_feature(train_df)
test_mf = get_mfcc_feature(test_df)

train_mel = get_feature_mel(train_df)
test_mel = get_feature_mel(test_df)

train_x = pd.concat([train_mel, train_mf], axis=1)
test_x = pd.concat([test_mel, test_mf], axis=1)

train_y = train_df['label']

# Create an AutoGluon dataset
train_x['label'] = train_df['label']
train_data = TabularDataset(train_x)
test_data = TabularDataset(test_x)

# train_data = TabularDataset(data=train_x)
train_data['label'] = train_y

# Specify the target column name
label_column = 'label'

# Create an AutoGluon predictor
predictor = TabularPredictor(label=label_column).fit(train_data)

# Use the predictor to generate a prediction
preds = predictor.predict(test_x)

submission = pd.read_csv('./_data/_dacon_emotion_recognition/sample_submission.csv')
submission['label'] = preds
submission.to_csv('./baseline_submission.csv', index=False)
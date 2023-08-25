import os
import scipy
# import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig
import librosa
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import csv
import math

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

DATASET_FOLDER = '../dataset'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame/'

FEATURES_FOLDER = 'myfeatures2'

ACTIVE_FOLDER = TRAIN_FOLDER

COMPUTE_EXTRACT = 0
COMPUTE_NORM = 1
COMPUTE_LABEL = 0

# max_i = 0
max_length = 2683092
max_length_c = 1440000 # 30sec*fs, fs=48k

# spectrogram for each channel 
nb_ch = 4
nfft = 2048
nb_bins = 64
hop_len = 480
fs = 48000
win_len = 960 # int(hop_len*fs)*2

if COMPUTE_EXTRACT :
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # 32k to convert from int to float -1 1

        q, r = divmod(mulchannel.shape[0], 1440000) # i want to keep features of 3000 since thats what the model can take in 
        
        mulaudio = []
        if q == 0:
            pad = np.zeros((max_length_c, 4))
            pad[:mulchannel.shape[0], :] = mulchannel
            mulaudio.append(pad)
        else: # audio is longer than 30secs
            mulchannel1 = mulchannel[:max_length_c, :]
            mulaudio.append(mulchannel1)

            pad = np.zeros((max_length_c, 4))
            mulchannel2 = mulchannel[(max_length_c+1):, :]
            pad[:mulchannel2.shape[0], :] = mulchannel2
            mulaudio.append(mulchannel1)
            mulaudio.append(pad)

        for cnt, audio in enumerate(mulaudio):
            spectra = np.zeros((3000, int(nfft/2) + 1, nb_ch), dtype=complex)
            
            for ch_cnt in range(nb_ch):
                stft_ch = librosa.core.stft(np.asfortranarray(audio[:, ch_cnt]), n_fft=nfft, hop_length=hop_len,
                                            win_length=win_len, window='hann')
                # print(audio.shape)
                # print(stft_ch.shape)
                spectra[:, :, ch_cnt] = stft_ch[:, :3000].T

            # gcc
            gcc_channels = nCr(spectra.shape[-1], 2)
            gcc_feat = np.zeros((spectra.shape[0], nb_bins, gcc_channels))
            cnt = 0
            for m in range(spectra.shape[-1]):
                for n in range(m+1, spectra.shape[-1]):
                    R = np.conj(spectra[:, :, m]) * spectra[:, :, n]
                    cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                    cc = np.concatenate((cc[:, -nb_bins//2:], cc[:, :nb_bins//2]), axis=-1)
                    gcc_feat[:, :, cnt] = cc
                    cnt += 1
            gcc_feat = gcc_feat.reshape((spectra.shape[0], nb_bins*gcc_channels))

            saved_file = file_name.split('.')[0] + '.csv'
            save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, saved_file)
            np.savetxt(save_path, gcc_feat, delimiter = ",")

# made a mistake and only took the input[max_length_c:, :] of the input 
# so label must be computed the same until i fix it

max_length = 350 # for a file of 1440000 samples (30 secs), with one label every 4096 samples, one label per 0.08secs
if COMPUTE_LABEL:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER))):
        print(file_cnt)
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER) + file_name
        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)

        out = np.zeros((max_length, 3))
        for x in f:
            if x[0] < 350:
                for y in x[1]:
                    out[x[0], :] = y[0][:350]    
        
        clean_file_name = file_name.split('.')[0] + '.csv'
        out_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label/') + clean_file_name 
        
        np.savetxt(out_file, out, delimiter = ",")

max = 0
min = 0
if COMPUTE_NORM: 
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label/'))):
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label/', file_name)
        print(file_cnt)
        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)

        max = f.max() if f.max() > max else max
        min = f.min() if f.min() < min else min

    print("max:", max)
    print("min:", min)

    print("normalising...")
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label/'))):
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label/', file_name)
        print(file_cnt)

        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)
        f = 2*(f-min)/(max-min) - 1
        
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER + '_label_norm/', file_name)
        np.savetxt(save_path, f, delimiter = ',')

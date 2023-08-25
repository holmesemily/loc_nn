import os
import scipy
import scipy.io.wavfile as wav
import scipy.signal as sig
import librosa
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import csv
import math
import matplotlib.pyplot as plt

DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame/'

FEATURES_FOLDER = 'features'

ACTIVE_FOLDER = TRAIN_FOLDER

COMPUTE_MEL = 0
COMPUTE_GCC = 1
COMPUTE_NORM = 0
COMPUTE_LABEL = 0

# spectrogram for each channel 
nb_ch = 4
nfft = 8192
nb_bins = 40
hop_len = 4092
win_len = 8192
fs = 48000

max = 0.0
min = 100.0

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

def compute_expression(Xi, Xj, time_delay_range):
    num_frequencies = Xi.shape[0]
    num_time_delays = 51
    
    result_temp = np.zeros((num_frequencies, num_time_delays), dtype=np.complex128)
    result = np.zeros((num_frequencies, num_time_delays), dtype=np.float128)

    for t_idx, tau in enumerate(time_delay_range):
        delay_factor = np.exp(1j * tau * time_delays)  # e^(jωτ) i think?
                
        numerator = Xi * np.conj(Xj) * delay_factor
        denominator = np.abs(Xi * np.conj(Xj))
        
        result_temp[:, t_idx] = np.real(numerator/denominator)

    result = np.nan_to_num(np.sum(result_temp, axis=0), copy=False) # sum over w
    return result

if COMPUTE_GCC:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # 32k to convert from int to float -1 1
        max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) # max frame in the gt file, 261 here
    
        spectra = np.zeros((max_frame, nfft//2 + 1, nb_ch), dtype=complex)
            
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(mulchannel[:, ch_cnt]), n_fft=nfft, hop_length=hop_len,
                                        win_length=win_len, window='hann')

            spectra[:, :, ch_cnt] = stft_ch[:, :max_frame].T # 261, 4097, 4

        time_delay_range = np.arange(-25, 26)   
        gcc_feat = np.zeros((max_frame, time_delay_range.shape[-1], nCr(spectra.shape[-1], 2)))

        for frame in range(max_frame):
            cnt = 0

            for m in range(spectra.shape[-1]):
                for n in range(m+1, spectra.shape[-1]):
                    sig1 = spectra[frame, :, m]
                    sig2 = spectra[frame, :, n]

                    # Precompute the time delays corresponding to the Fourier frequencies
                    nfft = sig1.shape[0]
                    freqs = np.fft.fftfreq(nfft, d=1/fs)
                    time_delays = np.fft.ifftshift(np.fft.fftfreq(nfft, d=1/fs))

                    # Compute the expression
                    expression_result = compute_expression(sig1, sig2, time_delay_range)
                    gcc_feat[frame, :, cnt] = expression_result
                    cnt += 1

        print(gcc_feat)
        print(gcc_feat.shape)
        # spectra = np.fft.fft(mulchannel)         # (length*fs, nb_ch) = (1077471,4)
        # max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) # max frame in the gt file, 261 here
        # time_delay_range = np.arange(-25, 26)    
        
        # gcc_feat = np.zeros((max_frame, time_delay_range.shape[-1], nCr(spectra.shape[-1], 2)))

        # for frame in range(max_frame):
        #     cnt = 0
        #     beg = frame*hop_len
        #     end = frame*hop_len+win_len

        #     for m in range(spectra.shape[-1]):
        #         for n in range(m+1, spectra.shape[-1]):
        #             sig1 = spectra[beg:end, m]
        #             sig2 = spectra[beg:end, n]

        #             # Precompute the time delays corresponding to the Fourier frequencies
        #             nfft = sig1.shape[0]
        #             freqs = np.fft.fftfreq(nfft, d=1/fs)
        #             time_delays = np.fft.ifftshift(np.fft.fftfreq(nfft, d=1/fs))

        #             # Compute the expression
        #             expression_result = compute_expression(sig1, sig2, time_delay_range)
        #             gcc_feat[frame, :, cnt] = expression_result
        #             cnt += 1

        # gcc_channels = nCr(spectra.shape[-1], 2)
        # gcc_feat = np.zeros((spectra.shape[0], nb_bins, gcc_channels))
        # cnt = 0
        # cc_temp = 0
        # for m in range(spectra.shape[-1]):
        #     for n in range(m+1, spectra.shape[-1]):
        #         # R = np.conj(spectra[:, :, m]) * spectra[:, :, n]
        #         # cc_temp += np.fft.irfft(np.exp(1.j*np.angle(R)))
        #         # print(cc_temp.shape)
        #         # cc = np.concatenate((cc_temp[:, -nb_bins//2:], cc_temp[:, :nb_bins//2]), axis=-1)
        #         gcc_feat[:, :, cnt] = cc
        #         cnt += 1

        # print(gcc_feat.shape)

        # check results

        num_frames, num_bins, num_channels = gcc_feat.shape
        for channel in range(num_channels):
            gcc_magnitude = np.abs(gcc_feat[:, :, channel])

            plt.figure(figsize=(10, 6))
            plt.imshow(gcc_magnitude.T, origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Time Frames')
            plt.ylabel('Frequency Bins')
            plt.title(f'GCC Feature Magnitude - Channel {channel}')
            plt.savefig("aaaaaaa.png")
            plt.cla()
            plt.clf()

        gcc_feat = gcc_feat.reshape((max_frame, time_delay_range.shape[-1]*nCr(spectra.shape[-1], 2)))
        print(np.max(gcc_feat))
        print(np.min(gcc_feat))
        saved_file = file_name.split('.')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, gcc_feat, delimiter = ",")

if COMPUTE_MEL :
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # 32k to convert from int to float -1 1
        
        max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) # max frame in the gt file

        mel_feat = np.zeros((max_frame, nb_bins, nb_ch))
        for ch_cnt in range(nb_ch):
            log_mel_spectra = librosa.feature.melspectrogram(sr=fs, y=mulchannel[:, ch_cnt], n_fft=8192, 
                                                             hop_length=4096, win_length=8192, window='hann', 
                                                             center=False, power=2.0, n_mels=nb_bins,
                                                             fmin = 0, fmax = 8000)
            mel_feat[:, :, ch_cnt] = log_mel_spectra[:, :max_frame].T

        mel_feat = mel_feat.reshape((mel_feat.shape[0]), nb_bins*nb_ch)
        max_temp = np.max(mel_feat)
        min_temp = np.min(mel_feat)
        max = max_temp if max_temp > max else max
        min = min_temp if min_temp < min else min
        print(max)
        print(min)

        # save feature 
        saved_file = file_name.split('.')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, mel_feat, delimiter = ",")

    # normalize 
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, file_name)
        print(file_cnt)
        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)

        f = (f-min)/(max-min) 

        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, file_name)
        np.savetxt(save_path, f, delimiter = ',')
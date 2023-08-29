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
import apkit

DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'
NORM_FOLDER = 'norm'
LABEL_FOLDER = 'label'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame/'
ALT_GCC_FOLDER = 'gcc2'
GCCFB_FOLDER = 'gccfb'

FEATURES_FOLDER = 'features'

ACTIVE_FOLDER = TEST_FOLDER

COMPUTE_MEL = 0
COMPUTE_GCC = 0
COMPUTE_GCC2 = 0
COMPUTE_GCC3 = 0
COMPUTE_GCCFB = 1
COMPUTE_NORM = 0
COMPUTE_LABEL = 0

# spectrogram for each channel 
nb_ch = 4
nfft = 8192
nb_bins = 51
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

def compute_azimuth(x,y):
    return (math.degrees(math.atan2((x),(y)))/360)+0.5 # normalise 0 1

if COMPUTE_GCC:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)

        check_file = file_name.split('.')[0] + '.csv'
        check_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, check_file)
        if os.path.isfile(check_file):
            continue
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # 32k to convert from int to float -1 1
        max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) # max frame in the gt file, 261 here

        spectra = np.zeros((max_frame, win_len//2 + 1, nb_ch), dtype=complex)
            
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(mulchannel[:, ch_cnt]), n_fft=win_len, hop_length=hop_len,
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

if COMPUTE_GCC2:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)

        check_file = file_name.split('.')[0] + '.csv'
        # check_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, check_file)
        # if os.path.isfile(check_file):
        #     continue
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # 32k to convert from int to float -1 1
        max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) # max frame in the gt file, 261 here

        spectra = np.zeros((max_frame, win_len//2 + 1, nb_ch), dtype=complex)
            
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(mulchannel[:, ch_cnt]), n_fft=win_len, hop_length=hop_len,
                                        win_length=win_len, window='hann')

            spectra[:, :, ch_cnt] = stft_ch[:, :max_frame].T # 261, 4097, 4

        time_delay_range = np.arange(-25, 26)   
        gcc_feat = np.zeros((max_frame, time_delay_range.shape[-1], nCr(spectra.shape[-1], 2)))

        gcc_channels = nCr(spectra.shape[-1], 2)
        gcc_feat = np.zeros((spectra.shape[0], nb_bins, gcc_channels))
        cnt = 0
        cc_temp = 0
        for m in range(spectra.shape[-1]):
            for n in range(m+1, spectra.shape[-1]):
                R = np.conj(spectra[:, :, m]) * spectra[:, :, n]
                cc_temp += np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc_temp[:, -nb_bins//2:], cc_temp[:, :nb_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1

        # # check results

        # num_frames, num_bins, num_channels = gcc_feat.shape
        # for channel in range(num_channels):
        #     gcc_magnitude = np.abs(gcc_feat[:, :, channel])

        #     plt.figure(figsize=(10, 6))
        #     plt.imshow(gcc_magnitude.T, origin='lower', aspect='auto', cmap='viridis')
        #     plt.colorbar(label='Magnitude')
        #     plt.xlabel('Time Frames')
        #     plt.ylabel('Frequency Bins')
        #     plt.title(f'GCC Feature Magnitude - Channel {channel}')
        #     plt.savefig("aaaaaaa.png")
        #     plt.cla()
        #     plt.clf()

        gcc_feat = gcc_feat.reshape((max_frame, 51*6))
        saved_file = file_name.split('.')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, gcc_feat, delimiter = ",")
   
if COMPUTE_GCCFB:
    # load signal
    win_size = 8192
    hop_size = 4096
    _FREQ_MAX = 8000
    _FREQ_MIN = 100
    nfbank = 40
    zoom = 25 # number of center coefficients on each side, excluding center
    eps = 1e-8

    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)
        # Weipeng He
        fs, sig = apkit.load_wav(cur_file)
        tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
        nch, nframe, _ = tf.shape

        # trim freq bins
        nfbin = _FREQ_MAX * win_size // fs  # 0-8kHz
        freq = np.fft.fftfreq(win_size)[:nfbin]
        tf = tf[:, :, :nfbin]

        # compute pairwise gcc on f-banks
        ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
        fbw = apkit.mel_freq_fbank_weight(nfbank,
                                        freq,
                                        fs,
                                        fmax=_FREQ_MAX,
                                        fmin=_FREQ_MIN)
        fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

        # merge to a single numpy array, indexed by 'tpbd'
        #                                           (time, pair, bank, delay)
        feature = np.asarray(
            [fbcc[(i, j)] for i in range(nch) for j in range(nch) if i < j])
        feature = np.moveaxis(feature, 2, 0)

        # and map [-1.0, 1.0] to 16-bit integer, to save storage space
        dtype = np.int16
        vmax = np.iinfo(dtype).max
        feature = (feature * vmax).astype(dtype)

        feature = feature.reshape((feature.shape[0], 6*40*51))
        saved_file = file_name.split('.')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, GCCFB_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, feature, delimiter = ",")
       

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

if COMPUTE_NORM:
    # normalize 
    max = 0.0
    min = 100.0
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, file_name.split('.')[0] + '.csv')
        print(file_cnt)
        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)

        min = np.min(f) if np.min(f) < min else min
        max = np.max(f) if np.max(f) > max else max

    print(max, min)

    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ACTIVE_FOLDER, file_name.split('.')[0] + '.csv')
        print(file_cnt)
        f = np.genfromtxt(cur_file, delimiter=',', skip_header=0)

        f = 2*(f-min)/(max-min)-1

        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, NORM_FOLDER, ACTIVE_FOLDER, file_name.split('.')[0] + '.csv')
        np.savetxt(save_path, f, delimiter = ',')

if COMPUTE_LABEL:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER))):
        print(file_cnt)
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER, file_name)
        lbl = np.load(cur_file, allow_pickle=True)

        ref_audio_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, NORM_FOLDER, ACTIVE_FOLDER, file_name.split('.')[0] + '.csv')
        max_len = np.genfromtxt(ref_audio_file, delimiter=',').shape[0]
        
        out_lbl = np.zeros((max_len, 1))
        for x in lbl:
            for y in x[1]:
                try:
                    out_lbl[x[0], 0] = compute_azimuth(y[0][0], y[0][1])
                finally: 
                    continue
        
        saved_file = file_name.split('.g')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, out_lbl, delimiter = ",")
        

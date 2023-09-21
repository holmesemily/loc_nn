"""
compute_features.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Simple feed-forward fully connected model for SSL estimations
"""

# Imports
import os
import scipy.io.wavfile as wav
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
DATASET_FOLDER = '../dataset/SSLR'
TRAIN_FOLDER = 'lsp_train_106'
TEST_FOLDER = 'lsp_test_106'
LABEL_FOLDER = 'label'
LABEL_C_FOLDER = 'label_c'

AUDIO_FOLDER = 'audio'
FEATURES_FOLDER = 'features'
NORM_FOLDER = 'norm'
GT_FOLDER = 'gt_frame/'
ALT_GCC_FOLDER = 'gcc'

ACTIVE_FOLDER = TRAIN_FOLDER

COMPUTE_MEL = 0
COMPUTE_GCC = 0
COMPUTE_LABEL = 0
COMPUTE_LABEL_C = 1
VISUALISE_RESULTS = 0

# Define Fourier and signal properties
nb_ch = 4
nfft = 8192
nb_bins = 51
hop_len = 4092
win_len = 8192
fs = 48000

# Combination formula
def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

# Azimuth from cartesian coordinates, scaled to [0, 1]
def compute_azimuth(x,y):
    return (math.degrees(math.atan2((x),(y)))/360) + 0.5 


'''
Input features are GCC-PHAT

Files are open sequencially. 
The final feature is a [X, 51, 6] 
            X frames
            51 bins
            6 combinations of microphone pairs

Output: [X, 51*6] .csv files of floats.
'''
if COMPUTE_GCC:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER))):
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, AUDIO_FOLDER, file_name)
        print(file_cnt)
        
        sample_freq, mulchannel = wav.read(cur_file) 
        mulchannel = mulchannel / 32768.0 # Convert to [-1, 1] float array

        # Frames for the ground truth labels are defined as
        # f = [f*HOP_SIZE, f*HOP_SIZE + WIN_SIZE]
        # Using the formula and audio length, we can deduce the number of frames of a given file
        max_frame = math.floor((mulchannel.shape[0] - win_len)/hop_len) 

        spectra = np.zeros((max_frame, win_len//2 + 1, nb_ch), dtype=complex)
            
        # Compute spectrogram with same parameters as dataset ground truth
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(mulchannel[:, ch_cnt]), n_fft=win_len, hop_length=hop_len,
                                        win_length=win_len, window='hann')

            spectra[:, :, ch_cnt] = stft_ch[:, :max_frame].T 

        gcc_channels = nCr(spectra.shape[-1], 2)
        gcc_feat = np.zeros((spectra.shape[0], nb_bins, gcc_channels))
        cnt = 0
        cc_temp = 0

        # Compute GCC-PHAT
        for m in range(spectra.shape[-1]):
            for n in range(m+1, spectra.shape[-1]):
                R = np.conj(spectra[:, :, m]) * spectra[:, :, n]
                cc_temp += np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc_temp[:, -nb_bins//2:], cc_temp[:, :nb_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1

    if VISUALISE_RESULTS:
        num_frames, num_bins, num_channels = gcc_feat.shape
        for channel in range(num_channels):
            gcc_magnitude = np.abs(gcc_feat[:, :, channel])

            plt.figure(figsize=(10, 6))
            plt.imshow(gcc_magnitude.T, origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Time Frames')
            plt.ylabel('Frequency Bins')
            plt.title(f'GCC Feature Magnitude - Channel {channel}')
            plt.savefig("gcc_feature_ch_" + str(channel) + ".png")
            plt.cla()
            plt.clf()

        # Export
        gcc_feat = gcc_feat.reshape((max_frame, 51*6))
        saved_file = file_name.split('.')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, ALT_GCC_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, gcc_feat, delimiter = ",")    


'''
Labels

Labels are an array of [X, 2]
            X frames
            2 columns: presence of an activity (1) or none (0)
                       azimuth normalised between 0 (0°) and 1 (360°)

                       0.5 (180°) means the sound is in front of the robott
'''
if COMPUTE_LABEL:
    max = 0.0
    min = 100.0
    
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER))):
        print(file_cnt)
        cur_file = os.path.join(DATASET_FOLDER, ACTIVE_FOLDER, GT_FOLDER, file_name)
        lbl = np.load(cur_file, allow_pickle=True)

        ref_audio_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, NORM_FOLDER, ACTIVE_FOLDER, file_name.split('.')[0] + '.csv')
        max_len = np.genfromtxt(ref_audio_file, delimiter=',').shape[0]
        
        out_lbl = np.zeros((max_len, 2))
        for x in lbl:
            for y in x[1]:
                try:
                    out_lbl[x[0], 1] = compute_azimuth(y[0][0], y[0][1])
                    out_lbl[x[0], 0] = 1 if out_lbl[x[0], 1] != 0 else 0
                finally: 
                    continue
        
        saved_file = file_name.split('.g')[0] + '.csv'
        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, ACTIVE_FOLDER, saved_file)
        np.savetxt(save_path, out_lbl, delimiter = ",")

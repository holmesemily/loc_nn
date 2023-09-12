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


'''
Labels (Classification)


'''
if COMPUTE_LABEL_C:
    for file_cnt, file_name in enumerate(os.listdir(os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, ACTIVE_FOLDER))):
        print(file_cnt)
        cur_file = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_FOLDER, ACTIVE_FOLDER, file_name)
        lbl = np.genfromtxt(cur_file, delimiter=',')

        # I don't know if a dict is the fastest way to compute it in Python, but, it works...
        degree_dict = {1: {0, 1, 2, 3, 4}, 2: {5, 6, 7, 8, 9}, 3: {10, 11, 12, 13, 14}, 4: {15, 16, 17, 18, 19}, 5: {20, 21, 22, 23, 24}, 6: {25, 26, 27, 28, 29}, 7: {32, 33, 34, 30, 31}, 8: {35, 36, 37, 38, 39}, 9: {40, 41, 42, 43, 44}, 10: {45, 46, 47, 48, 49}, 11: {50, 51, 52, 53, 54}, 12: {55, 56, 57, 58, 59}, 13: {64, 60, 61, 62, 63}, 
                    14: {65, 66, 67, 68, 69}, 15: {70, 71, 72, 73, 74}, 16: {75, 76, 77, 78, 79}, 17: {80, 81, 82, 83, 84}, 18: {85, 86, 87, 88, 89}, 19: {90, 91, 92, 93, 94}, 20: {96, 97, 98, 99, 95}, 21: {100, 101, 102, 103, 104}, 22: {105, 106, 107, 108, 109}, 23: {110, 111, 112, 113, 114}, 24: {115, 116, 117, 118, 119}, 25: {120, 121, 122, 123, 124},
                    26: {128, 129, 125, 126, 127}, 27: {130, 131, 132, 133, 134}, 28: {135, 136, 137, 138, 139}, 29: {140, 141, 142, 143, 144}, 30: {145, 146, 147, 148, 149}, 31: {150, 151, 152, 153, 154}, 32: {155, 156, 157, 158, 159}, 33: {160, 161, 162, 163, 164}, 34: {165, 166, 167, 168, 169}, 35: {170, 171, 172, 173, 174}, 36: {175, 176, 177, 178, 179}, 
                    37: {180, 181, 182, 183, 184}, 38: {185, 186, 187, 188, 189}, 39: {192, 193, 194, 190, 191}, 40: {195, 196, 197, 198, 199}, 41: {200, 201, 202, 203, 204}, 42: {205, 206, 207, 208, 209}, 43: {210, 211, 212, 213, 214}, 44: {215, 216, 217, 218, 219}, 45: {224, 220, 221, 222, 223}, 46: {225, 226, 227, 228, 229}, 47: {230, 231, 232, 233, 234}, 
                    48: {235, 236, 237, 238, 239}, 49: {240, 241, 242, 243, 244}, 50: {245, 246, 247, 248, 249}, 51: {250, 251, 252, 253, 254}, 52: {256, 257, 258, 259, 255}, 53: {260, 261, 262, 263, 264}, 54: {265, 266, 267, 268, 269}, 55: {270, 271, 272, 273, 274}, 56: {275, 276, 277, 278, 279}, 57: {280, 281, 282, 283, 284}, 58: {288, 289, 285, 286, 287}, 
                    59: {290, 291, 292, 293, 294}, 60: {295, 296, 297, 298, 299}, 61: {300, 301, 302, 303, 304}, 62: {305, 306, 307, 308, 309}, 63: {310, 311, 312, 313, 314}, 64: {315, 316, 317, 318, 319}, 65: {320, 321, 322, 323, 324}, 66: {325, 326, 327, 328, 329}, 67: {330, 331, 332, 333, 334}, 68: {335, 336, 337, 338, 339}, 69: {340, 341, 342, 343, 344}, 
                    70: {345, 346, 347, 348, 349}, 71: {352, 353, 354, 350, 351}, 72: {355, 356, 357, 358, 359}}
        
        lbl_new = np.zeros((lbl.shape[0], 1))
        for index, vals in enumerate(lbl):
            if vals[0] == 1:
                lbl_new[index] = next((key for key, degree_set in degree_dict.items() if int(vals[1]*360) in degree_set), None)

        save_path = os.path.join(DATASET_FOLDER, FEATURES_FOLDER, LABEL_C_FOLDER, ACTIVE_FOLDER, file_name)
        np.savetxt(save_path, lbl_new, delimiter = ",", fmt='%i')
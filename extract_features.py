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

# constants
directory = '../dataset/foa_dev'
out_directory = '../dataset/myfeatures/foa_dev'
desc_directory = '../dataset/metadata_dev'
out_directory_label = '../dataset/myfeatures/foa_dev_label'

nb_channels = 4
fs=24000
eps = 1e-8
hop_len = 0.02
# nfft = (2**(int(hop_len*fs)-1)).bit_length()
nfft = 2048
win_len = int(hop_len*fs)*2
nb_mel_bins = 64
max_frames = 3000
max_label_frames = 600

def preprocess_features():
# Setting up folders and filenames
    feat_dir = '../dataset/myfeatures/foa_dev'
    feat_dir_norm = '../dataset/myfeatures/foa_dev_norm3'
    normalized_features_wts_file = '../dataset/myfeatures/foa_wts'
    spec_scaler = None

    # pre-processing starts
    print('Estimating weights:')

    spec_scaler = preprocessing.StandardScaler()
    for file_cnt, file_name in enumerate(os.listdir(feat_dir)):
        print('{}: {}'.format(file_cnt, file_name))
        # feat_file = np.load(os.path.join(feat_dir, file_name))
        feat_file = pd.read_csv(os.path.join(feat_dir, file_name), header=None, encoding = "ISO-8859-1")
        print(feat_file.shape)
        spec_scaler.partial_fit(feat_file)
        del feat_file
    joblib.dump(
        spec_scaler,
        normalized_features_wts_file
    )
    print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

    print('Normalizing feature files:')
    for file_cnt, file_name in enumerate(os.listdir(feat_dir)):
        print('{}: {}'.format(file_cnt, file_name))
        # feat_file = np.load(os.path.join(feat_dir, file_name))
        feat_file = pd.read_csv(os.path.join(feat_dir, file_name), header=None, encoding = "ISO-8859-1")
        feat_file = spec_scaler.transform(feat_file)
        # np.save(
        #     os.path.join(feat_dir_norm, file_name),
        #     feat_file
        # )
        feat_file_df = pd.DataFrame(feat_file)
        feat_file_df.to_csv(os.path.join(feat_dir_norm, file_name), encoding = "ISO-8859-1")
        del feat_file

    print('normalized files written to {}'.format(feat_dir_norm))

def extract_labels():
    print("Extracting labels")
    for filename in os.listdir(out_directory_label):
        slabel_mat = np.zeros((max_label_frames, 2))
        cur_file = os.path.join(out_directory_label, filename)
        desc_l = np.genfromtxt(cur_file, delimiter=',', dtype=int)
        print(filename)
        for index in range(len(desc_l)):         # for every line in input file
            # print("à l'index:", index, "il y a ", desc_l[index, :])
            slabel_mat[desc_l[index, 0], 0] = desc_l[index, 1]
            slabel_mat[desc_l[index, 0], 1] = desc_l[index, 2]
            # print(slabel_mat[desc_l[index, 0], :])
        print(slabel_mat)
        np.savetxt(os.path.join(out_directory_label, filename), slabel_mat, delimiter = ",", fmt='%0u')      
        # slabel_mat_df = pd.DataFrame(slabel_mat)
        # slabel_mat_df.to_csv(os.path.join(out_directory_label, filename))


print("Generating spectrograms...")
for filename in os.listdir(directory):
    print(filename)
    cur_file = os.path.join(directory, filename)
    
    # get input audio, make sure its 60*24000 samples
    sample_freq, mulchannel = wav.read(cur_file) 

    mulchannel = mulchannel[:, :4] / 32768.0 + eps
    if mulchannel.shape[0] < 60*fs:
        zero_pad = np.random.rand(60 - mulchannel.shape[0], mulchannel.shape[1])*eps
        mulchannel = np.vstack((mulchannel, zero_pad))
    elif mulchannel.shape[0] > 60*fs:
        mulchannel = mulchannel[:60, :]

    # spectrogram for each channel 
    nb_ch = mulchannel.shape[1]
    nb_bins = nfft//2
    spectra = np.zeros((3000, nb_bins + 1, nb_ch), dtype=complex)
    for ch_cnt in range(nb_ch):
        print(mulchannel.shape)
        stft_ch = librosa.core.stft(np.asfortranarray(mulchannel[:, ch_cnt]), n_fft=nfft, hop_length=int(hop_len*24000),
                                    win_length=win_len, window='hann')
        print(stft_ch.shape)
        spectra[:, :, ch_cnt] = stft_ch[:, :3000].T

#     # mel 
#     mel_wts = librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=nb_mel_bins).T
#     mel_feat = np.zeros((spectra.shape[0], nb_mel_bins, spectra.shape[-1]))

#     for ch_cnt in range(spectra.shape[-1]):
#         mag_spectra = np.abs(spectra[:, :, ch_cnt])**2
#         mel_spectra = np.dot(mag_spectra, mel_wts)
#         log_mel_spectra = librosa.power_to_db(mel_spectra)
#         mel_feat[:, :, ch_cnt] = log_mel_spectra
#         # print("mel feat shape", mel_feat.shape)
#         # print(mel_feat[1][0][1], mel_feat[1][1][1], mel_feat[1][2][1])
#     mel_feat = mel_feat.reshape((spectra.shape[0], nb_mel_bins * spectra.shape[-1]))
#     # print("mel feat shape after formatting", mel_feat.shape)
#     # print(mel_feat[1][0], mel_feat[1][1], mel_feat[1][2])

#     df_feat = pd.DataFrame(mel_feat)
#     filename_clean = os.path.splitext(filename)[0] + ".csv"
#     df_feat.to_csv(os.path.join(out_directory, filename_clean))

# preprocess_features()
# extract_labels()


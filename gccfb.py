#!/usr/bin/env python
"""
extract_cov_mat.py
Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import argparse

import numpy as np

import apkit

_WAVE_SUFFIX = '.wav'
_NPY_SUFFIX = '.npy'

_FREQ_MAX = 8000
_FREQ_MIN = 100


def main(wavfile, destfile, win_size, hop_size, nfbank, zoom, eps):
    # load signal
    fs, sig = apkit.load_wav(wavfile)
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
    nch, nframe, _ = tf.shape

    # trim freq bins
    nfbin = _FREQ_MAX * win_size / fs  # 0-8kHz
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

    np.save(destfile, feature)

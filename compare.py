'''
compare.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Compare the predictions with the ground values for a given file
'''

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy, math

# Imports 
pred_detec = '../dataset/SSLR/predict/pred_detec.csv'
pred_doa = '../dataset/SSLR/predict/pred_doa.csv'
gt_file = '../dataset/SSLR/features/label_c/lsp_train_106/ssl-data_2017-05-13-15-25-43_0.w8192_o4096.csv'


# Get average per chunk of voice activity
def AverageByGroup(df):
    result_data = []
    group_values = []

    for val in df:
        if val != 0:
            group_values.append(val)
        else:
            if group_values:
                avg_val = group_values.copy()
                if len(avg_val) > 2:
                    avg_val.remove(max(avg_val))
                    avg_val.remove(min(avg_val))
                average = sum(avg_val) / len(avg_val)
                result_data.extend([average] * len(group_values))
                group_values = []
            result_data.append(0)

    if group_values:
        average = sum(group_values) / len(group_values)
        result_data.extend([average] * len(group_values))
    return result_data

# interpolate by a given coefficient. coef = 2 will return an array
# with twice as many points
def Interpolate(array, coef):
    x = np.linspace(0, array.shape[0], array.shape[0])
    f = scipy.interpolate.interp1d(x, array)

    xs = np.linspace(0, array.shape[0], array.shape[0]*coef)
    interp = f(xs)
    
    return xs, interp

def Post_Proc(array):
    kernel_size = 50
    kernel = np.ones(kernel_size) / kernel_size 
    data_convolved = np.convolve(array, kernel, mode='same')
    return data_convolved

def accuracy(gt, exp, delta):
    res = abs(gt - exp) < (delta)
    return res.sum()/res.shape[0]

# Create figure
fig, axs = plt.subplots(1, 1, figsize=(8, 5)) 

# Predictions
df_pred_doa = np.genfromtxt(pred_doa, delimiter=',')
df_pred_doa = np.argmax(df_pred_doa, axis=1)
print(df_pred_doa)

# Ground truths
df_gt = np.genfromtxt(gt_file, delimiter=',')

x = np.linspace(0, df_pred_doa.shape[0], df_pred_doa.shape[0])

plt.plot(x, df_gt, color='blue', linestyle='-', label='ground truth')
plt.plot(x, df_pred_doa, color='red', linestyle=':', linewidth = 1, label='prediction')

axs.set_title('Prediction of Azimuth')
axs.legend()
axs.set_ylim(-5, 73)
plt.xlabel("frame number")
plt.ylabel("class name (5° increment between class)")

plt.savefig("img/comparison.png")

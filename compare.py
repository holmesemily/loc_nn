'''
compare.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Compare the predictions with the ground values for a given file
'''

import numpy as np
from matplotlib import pyplot as plt
import scipy

# Imports 
pred_detec = '../dataset/SSLR/predict/pred_detec.csv'
pred_doa = '../dataset/SSLR/predict/pred_doa.csv'
gt_file = '../dataset/SSLR/features/label/lsp_train_106/ssl-data_2017-05-13-15-25-43_0.w8192_o4096.csv'


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

# Post process by using a smoothing filter
def Post_Proc(array):
    kernel_size = 50
    kernel = np.ones(kernel_size) / kernel_size 
    data_convolved = np.convolve(array, kernel, mode='same')
    return data_convolved

def accuracy(gt, exp, delta):
    res = abs(gt - exp) < (delta)
    return res.sum()/res.shape[0]

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(16, 5)) 


# Predictions
df_pred_detec = np.genfromtxt(pred_detec, delimiter=',')
df_pred_doa = np.genfromtxt(pred_doa, delimiter=',')

# Ground truths
df_gt = np.genfromtxt(gt_file, delimiter=',')
df_gt_detec = df_gt[:, 0]
df_gt_doa = df_gt[:, 1]

x = np.linspace(0, df_pred_doa.shape[0], df_pred_doa.shape[0])

threshold = 0.5      # Threshold to push values towards 0 or 1
df_pred_detec = (df_gt_detec >= threshold).astype(int)

axs[0].plot(x, df_gt_doa*360, color='blue', linestyle='-', label='ground truth')
axs[0].plot(x, df_pred_doa*360, color='green', linestyle=':', linewidth = 1, label='prediction')

# Post-process with the detection network 
df_pred_doa[df_pred_detec == 0] = 0
averaged_pred_doa = AverageByGroup(df_pred_doa*360)
axs[0].plot(x, averaged_pred_doa, color='red', linestyle='-', label='averaged prediction')

# Post-process with moving average
# axs[0].plot(df_pred_doa.rolling(window=5).mean()*360, color='red', linestyle='-', label='averaged prediction')

axs[0].set_title('Prediction of Azimuth based on Detection of a Source')
axs[0].legend()
axs[0].set_ylim(-5, 360)

# Detection network output, if it applies
# axs[1].plot(df_gt_detec, color='blue', linestyle='-', label='ground truth')
# axs[1].plot(df_pred_detec, color='red', linestyle='-', label='prediction')
# axs[1].set_title('Prediction of Detection of a Source')
# axs[1].legend()

degree = 10
print("accuracy:", accuracy(df_gt[:, 1], df_gt_detec, degree/360))

plt.tight_layout()
plt.savefig("img/comparison.png")

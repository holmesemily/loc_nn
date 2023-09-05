'''
compare.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Compare the predictions with the ground values for a given file
'''

import pandas as pd
import os
from matplotlib import pyplot as plt


# Imports 
pred_detec = '../dataset/SSLR/predict/pred_detec.csv'
pred_doa = '../dataset/SSLR/predict/pred_doa.csv'
gt_file = '../dataset/SSLR/features/label2/lsp_train_106/ssl-data_2017-05-13-15-25-43_0.w8192_o4096.csv'


# Get average per chunk of voice activity
def AverageByGroup(df):
    result_data = []
    group_values = []

    for val in df.iloc[:, 0]:
        if val != 0:
            group_values.append(val)
        elif group_values:
            result_data.append(0)
            average = sum(group_values) / len(group_values)
            result_data.extend([average] * len(group_values))
            group_values = []
        else:
            result_data.append(0)

    if group_values:
        average = sum(group_values) / len(group_values)
        result_data.extend([average] * len(group_values))

    return pd.DataFrame(result_data)


# Create figure
fig, axs = plt.subplots(1, 2, figsize=(16, 5)) 


# Predictions
df_pred_detec = pd.read_csv(pred_detec, header=None)
df_pred_doa = pd.read_csv(pred_doa, header=None)

# Ground truths
df_gt = pd.read_csv(gt_file, header=None)
df_gt_detec = df_gt.iloc[:, 0]
df_gt_doa = df_gt.iloc[:, 1]

threshold = 0.5      # Threshold to push values towards 0 or 1
df_pred_detec = df_gt_detec.apply(lambda x: 1 if x >= threshold else 0)
df_pred_doa.loc[df_pred_detec == 0] = 0

axs[0].plot(df_gt_doa*360, color='blue', linestyle='-', label='ground truth')

axs[0].plot(df_pred_doa*360, color='red', linestyle=':', linewidth = 1, label='prediction')
axs[0].plot(AverageByGroup(df_pred_doa)*360, color='red', linestyle='-', label='averaged prediction')
axs[0].set_title('Prediction of Azimuth based on Detection of a Source')
axs[0].legend()
axs[0].set_ylim(-5, 360)

axs[1].plot(df_gt_detec, color='blue', linestyle='-', label='ground truth')
axs[1].plot(df_pred_detec, color='red', linestyle='-', label='prediction')
axs[1].set_title('Prediction of Detection of a Source')
axs[1].legend()

plt.tight_layout()
plt.savefig("img/comparison.png")
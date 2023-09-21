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
gt_file = '../dataset/SSLR/features/label_c/lsp_train_106/ssl-data_2017-05-13-15-25-43_4.w8192_o4096.csv'


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

# Interpolate by a given coefficient. coef = 2 will return an array
# with twice as many points
def Interpolate(array, coef):
    x = np.linspace(0, array.shape[0], array.shape[0])
    f = scipy.interpolate.interp1d(x, array)

    xs = np.linspace(0, array.shape[0], array.shape[0]*coef)
    interp = f(xs)
    
    return xs, interp


# Computes accuracy with a specific delta   
def accuracy(gt, exp, delta):
    res = abs(gt - exp) < (delta)
    return res.sum()/res.shape[0]

# Post-processing for classification - spurious value removal
def Post_Proc_C(array):
    for index in range(len(array)):
        if index != 0 and index != (len(array)-1):
            if array[index-1] == array[index+1] and array[index] != array[index-1]: # if odd value sandwiched between two identical values, change it to that identical value
                array[index] = array[index-1]
    return array

# Create figure
fig, axs = plt.subplots(1, 1, figsize=(8, 5)) 

# Predictions
df_pred_doa = np.genfromtxt(pred_doa, delimiter=',')
df_pred_doa = np.argmax(df_pred_doa, axis=1)   

# Ground truths
df_gt = np.genfromtxt(gt_file, delimiter=',')

x = np.linspace(0, df_pred_doa.shape[0], df_pred_doa.shape[0])

pred_doa = 'pred_doa_toedit_beforepp.csv'
df_pred_doa = np.genfromtxt(pred_doa, delimiter=',')

plt.plot(x, df_gt, color='blue', linestyle='-', label='ground truth')
plt.plot(x, df_pred_doa, color='red', linestyle=':', linewidth = 1, label='prediction')
plt.plot(x, Post_Proc_C(df_pred_doa), color='red', linestyle='-', linewidth = 1, label='post-processed prediction')

axs.set_title('Prediction of Azimuth')
axs.legend()
axs.set_ylim(-5, 73)
plt.xlabel("frame number")
plt.ylabel("class name (5° increment between class)")

plt.savefig("img/comparison.png")

degree = 2
print("accuracy:", accuracy(df_gt, Post_Proc_C(df_pred_doa), degree))
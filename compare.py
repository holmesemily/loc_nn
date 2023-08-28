#%% 
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [14.00, 7.0]
plt.rcParams["figure.autolayout"] = True

min_i = 28.056959861353036
max_i = -73.14660305467869

pred_file = '../dataset/SSLR/predict/pred.csv'
gt_file = '../dataset/SSLR/features/label/lsp_train_106/ssl-data_2017-05-13-15-25-43_0.w8192_o4096.csv'
df_pred = pd.read_csv(pred_file, header=None)
df_gt = pd.read_csv(gt_file, header=None)

plt.plot(df_pred, color='green', linestyle=':')
plt.plot(df_gt, color='blue', linestyle='-')
plt.plot(df_pred.rolling(window=5).mean(), color='red', linestyle='-')
plt.savefig("aaaaaaa.png")
# %%

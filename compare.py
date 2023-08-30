
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [14.00, 7.0]
plt.rcParams["figure.autolayout"] = True


def AverageByGroup(df):
    result_data = []
    group_values = []

    for val in df.iloc[:, 0]:
        if val != 0:
            group_values.append(val)
        elif group_values:
            average = sum(group_values) / len(group_values)
            result_data.extend([average] * len(group_values))
            group_values = []
        else:
            result_data.append(0)

    if group_values:
        average = sum(group_values) / len(group_values)
        result_data.extend([average] * len(group_values))

    return pd.DataFrame(result_data)

fig, axs = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns

pred_detec = '../dataset/SSLR/predict/pred_detec.csv'
pred_doa = '../dataset/SSLR/predict/pred_doa.csv'
gt_file = '../dataset/SSLR/features/label/lsp_train_106/ssl-data_2017-05-13-15-25-43_0.w8192_o4096.csv'
df_pred_detec = pd.read_csv(pred_detec, header=None)
df_pred_doa = pd.read_csv(pred_doa, header=None)
df_gt = pd.read_csv(gt_file, header=None)
df_gt_detec = df_gt.iloc[:, 0]
df_gt_doa = df_gt.iloc[:, 1]

threshold = 0.5
df_pred_detec = df_gt_detec.apply(lambda x: 1 if x >= threshold else 0)

df_pred_doa.loc[df_pred_detec == 0] = 0

# plt.plot(df_pred, color='red',  linestyle=':')
axs[0].plot(df_gt_doa, color='blue', linestyle='-')
# axs[0].plot(df_pred_doa.rolling(window=3).mean(), color='red', linestyle='-')
axs[0].plot(df_pred_doa, color='red', linestyle=':', linewidth = 1)
axs[0].plot(AverageByGroup(df_pred_doa), color='red', linestyle='-')

axs[1].plot(df_gt_detec, color='blue', linestyle='-')
axs[1].plot(df_pred_detec, color='red', linestyle='-')
plt.tight_layout()
plt.savefig("img/aaaaaaaa.png")
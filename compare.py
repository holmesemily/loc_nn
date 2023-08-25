#%% 
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

min_i = 28.056959861353036
max_i = -73.14660305467869

dump_dir = '../dataset/results'
og_dir = '../dataset/myfeatures2/lsp_train_106_label_norm'
og_file = os.path.join(og_dir, "ssl-data_2017-05-13-15-25-43_0.csv")
dump_file = os.path.join(dump_dir, "model_cnn_newdataset_predict_0.csv")
# dump_file2 = os.path.join(dump_dir, "model_cnn_newnorm_predict_0.csv")
df_dump = pd.read_csv(dump_file, header=None)
df_og = pd.read_csv(og_file, header=None)

plt.plot(df_dump.iloc[:,2], color='green')
plt.plot(df_og.iloc[:,2], color='blue')
plt.savefig("aaaaaaa.png")
# %%

import numpy as np
import os

# remove files with overlapping sources from SSLR dataset

TRAIN_FOLDER = '../dataset/SSLR/lsp_train_106'
TEST_FOLDER = '../dataset/SSLR/lsp_test_106'

AUDIO_FOLDER = 'audio'
GT_FOLDER = 'gt_frame'

FEATURES_FOLDER = '../dataset/SSLR/myfeatures2'
TRAIN_FEATURES = 'trainfeatures'

for file_cnt, file_name in enumerate(os.listdir(os.path.join(TRAIN_FOLDER, GT_FOLDER))):
    cur_file = os.path.join(TRAIN_FOLDER, GT_FOLDER, file_name)
    audio_file_name = file_name.split('.')[0] + ".wav"
    audio_file = os.path.join(TRAIN_FOLDER, AUDIO_FOLDER, audio_file_name)
    print(cur_file)
    f = np.load(cur_file, allow_pickle=True)
    for x in f:
        if len(x[1]) > 1: # more than 1 element in the tuple -> overlapping
            os.remove(cur_file)
            os.remove(audio_file) #remove the corresponding audio. i ignore the gt_file since i don't use it
            break
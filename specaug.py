"""
specaug.py

Written by Emily Holmes during a LAAS-CNRS internship, 2023
Implementation of data augmentation following the time warping from the SpecAugment method by DS Park et al. 
"""

import numpy as np
import os 
import random

dir_dev = '../dataset/myfeatures/path'      # path to normalised features
dir_label = '../dataset/myfeatures/path'    # path for normalised labels

for filename in os.listdir(dir_dev):
    print(filename)
    cur_file_dev = os.path.join(dir_dev, filename)
    cur_file_label = os.path.join(dir_label, filename)
    
    f_dev = np.genfromtxt(cur_file_dev, delimiter=',')
    f_label = np.genfromtxt(cur_file_label, delimiter=',')
        
    shift_amt = random.randrange(60)

    # output files
    foa_aug = np.zeros((3000, 256))
    label_aug = np.zeros((600, 2))
    
    # shift features left by shift_amt - time warping
    foa_aug[:3000-(shift_amt*50), :] = f_dev[shift_amt*50:, :]
    foa_aug[3000-(shift_amt*50):, :] = f_dev[:shift_amt*50, :]
    
    label_aug[:600-(shift_amt*10), :] = f_label[shift_amt*10:, :]
    label_aug[600-(shift_amt*10):, :] = f_label[:shift_amt*10, :]
    
    np.savetxt(os.path.join(dir_dev, filename.split()[0] + "_aug.csv"), foa_aug, delimiter = ",")  
    np.savetxt(os.path.join(dir_label, filename.split()[0] + "_aug.csv"), label_aug, delimiter = ",")  
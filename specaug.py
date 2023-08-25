import numpy as np
import os 
import random

dir_dev = '../dataset/myfeatures/foa_dev_norm3'
dir_label = '../dataset/myfeatures/foa_dev_label_norm2'

for filename in os.listdir(dir_dev):
    print(filename)
    cur_file_dev = os.path.join(dir_dev, filename)
    cur_file_label = os.path.join(dir_label, filename)
    
    f_dev = np.genfromtxt(cur_file_dev, delimiter=',')
    f_label = np.genfromtxt(cur_file_label, delimiter=',')
    
    # foa_aug = np.zeros((600, 256))
    # label_aug = np.zeros((1200, 2))
    
    # for index in range(0, 298, 2):
    #     foa_aug[index, :]   = f_dev[index, :]
    #     foa_aug[index+1, :] = f_dev[index, :]
        
    # for index in range(0, 598, 2):
    #     label_aug[index+1, :] = f_label[index, :]
    #     label_aug[index+1, :] = f_label[index, :]
        
    shift_amt = random.randrange(60)
    foa_aug = np.zeros((3000, 256))
    label_aug = np.zeros((600, 2))
    
    # shift features left by shift_amt - time warping
    foa_aug[:3000-(shift_amt*50), :] = f_dev[shift_amt*50:, :]
    foa_aug[3000-(shift_amt*50):, :] = f_dev[:shift_amt*50, :]
    
    label_aug[:600-(shift_amt*10), :] = f_label[shift_amt*10:, :]
    label_aug[600-(shift_amt*10):, :] = f_label[:shift_amt*10, :]
    
    # foa_aug = np.reshape(foa_aug, (300,64,4))
    
    # mask_size = random.randrange(5)
    # # for each channel, do freq and time masking - i dont know if its good
    # for index in range(2):
    #     shift_amt_freq = random.randrange(64-mask_size)  # make sure we dont exceed bounds
    #     shift_amt_time = random.randrange(300-mask_size) # make sure we dont exceed bounds
        
    #     foa_aug[shift_amt_time:shift_amt_time+mask_size, :, index] = 0
    #     foa_aug[:, shift_amt_freq:shift_amt_freq+mask_size, index] = 0
    
    np.savetxt(os.path.join(dir_dev, filename.split()[0] + "_aug.csv"), foa_aug, delimiter = ",")  
    np.savetxt(os.path.join(dir_label, filename.split()[0] + "_aug.csv"), label_aug, delimiter = ",")  
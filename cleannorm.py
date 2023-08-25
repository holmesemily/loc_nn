import numpy as np
import os 

dir = '../dataset/myfeatures/foa_dev_norm3'
dir_label = '../dataset/myfeatures/foa_dev_label'

outdir = '../dataset/myfeatures/foa_dev_norm3'
outdir_label = '../dataset/myfeatures/foa_dev_label_norm2'

i=0
max=28.056959861353036
min=-73.14660305467869

min_o_1 = -180.0
max_o_1 = 180.0
min_o_2 = -40.0
max_o_2 = 40.0

for filename in os.listdir(dir):
    print(i)
    i += 1
    cur_file = os.path.join(dir, filename)
    f = np.genfromtxt(cur_file, delimiter=',', skip_header=0, dtype=float)
    # print(f.shape)
    if f.shape != (3000, 256): 
        f = np.delete(f, 0, 1)
        f = np.delete(f, 0, 0)
    # f = np.delete(f, 0, 1
    # print(f.shape)
    # max = f.max() if f.max() > max else max
    # min = f.min() if f.min() < min else min
    # temp_xmax, temp_ymax = f.max(axis=0)
    # temp_xmin, temp_ymin = f.min(axis=0)
    # x_max = temp_xmax if temp_xmax > x_max else x_max
    # x_min = temp_xmin if temp_xmin < x_min else x_min
    # y_max = temp_ymax if temp_ymax > y_max else y_max
    # y_min = temp_ymin if temp_ymin < y_min else y_min
    # print("max=",max)
    # print("min=",min)
    np.savetxt(cur_file, f, delimiter = ",")  
    # np.savetxt(cur_file, f, fmt='%i', delimiter = ",")  


# print("Total x_max:", x_max)
# print("Total x_min:", x_min)
# print("Total y_max:", y_max)
# print("Total y_min:", y_min)
i=0
mean=(max-min)/2

# #TODO need to save normalisation (min max) to scale future data the same
# for filename in os.listdir(dir):
#     print(i)
#     i += 1
#     cur_file = os.path.join(dir, filename)
#     out_file = os.path.join(outdir, filename)
#     f = np.genfromtxt(cur_file, delimiter=',', skip_header=0, dtype=float)
#     # f between min and max
#     # f[:,0] = f[:,0]/180
#     # f[:,1] = f[:,1]/40
#     f = (f + mean)/mean
#     print(f)
#     np.savetxt(out_file, f, delimiter = ",")  

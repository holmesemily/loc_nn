import numpy as np
import os

directory = '../dataset/metadata_dev'
out_dir = '../dataset/myfeatures/foa_dev_label'
i = 0
for filename in os.listdir(directory):
    print(i)
    i += 1
    cur_file = os.path.join(directory, filename)
    f = np.genfromtxt(cur_file, delimiter=',')
    # f = np.delete(f, 1, 1)
    # print(f)
    np.savetxt(os.path.join(out_dir, filename), f.astype(int), fmt='%0u', delimiter = ",")  
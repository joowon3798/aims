import os
import sys
import random
from make_data import *

input_dir = '/data/aims/STEM_Defect_Analysis_Data/'
save_dir = '/data/aims/STEM_Defect_Analysis_Data/'

# Get current working directory
curDir = os.getcwd()

# Change directory to input_dir to select subdirectories to divide training & testing set
os.chdir(input_dir)

# List all subdirectories
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Total number of subdirectories
tot = len(all_subdirs)

# Number of testing directories to choose from
k = round(tot*0.8) # Just to ensure that nth funny is happening with round()

# Train and test directories
# set seed for reproducibility
random.seed(1)
train_dirs = random.sample(all_subdirs, k) # 16
test_dirs = [t for t in all_subdirs if t not in train_dirs] # 4

# Change directory back to previous directory
os.chdir(curDir)

'''
train_dirs = ["0", "12", "20", "17"]
test_dirs = ["8"]
'''
# total label_list = ["1vacancy", "2vacancy"]
label_list = ["1vacancy"]
parsed_dir_name = 'parsed_label_1vacancy'
ftype = '.tiff'
 
l_shape = (256, 256)
stride = (36, 36)

one_pickle=False 
tr_fsize = 2000
ts_fsize = 400

tol = 0.05

ones_percent = 0

'''
Note on creating augments:
1. While there are multiple level of magnifications that you can use,
I personally decided to use no magnification as the resulting images are pixelated and not of good quality. Thus, the post_process file is set. are not using magnification (only mag=0), 

2. The original make_data method will only create augments for folders that will get used as train sets. However, since 1_preprocess.py randomly selects train folders and test folders, always run create_augments for all folders before running make_data. Otherwise, make_data will throw an error if a folder without augments (=folder previously selected as test folder when creating a different imageset) is selected as a train folder.
'''

# Create autments for the files in input_dir
create_augments(input_dir, train_dirs, ftype)

# Create the picke files using the files from input_dir
# Train file(s)
make_data(input_dir, save_dir, train_dirs, label_list, l_shape, stride, ftype, parsed_dir_name=parsed_dir_name, \
        prefix="train", AUG=False, tol=tol, ones_pcent=ones_percent, one_save=one_pickle, fsize=tr_fsize)

# Test file
make_data(input_dir, save_dir, test_dirs, label_list, l_shape, stride, ftype, parsed_dir_name=parsed_dir_name, \
        prefix="test", AUG=False, tol=tol, ones_pcent=ones_percent, one_save=one_pickle, fsize=ts_fsize)

import numpy as np

parsed_fn = save_dir + parsed_dir_name + "/test_00000.p"

# Select a random image from the test file and print it out to the screen
check_data(parsed_fn, l_shape=l_shape)


# Write text files with info on which images were used for training/testing
# For record-keeping only, no other use in model training
sys.stdout = open("b1_combined82_SV1.txt", "w")
print("Training directories are: ")
print(*train_dirs)
print("\nTesting directories are: ")
print(*test_dirs)
sys.stdout.close()

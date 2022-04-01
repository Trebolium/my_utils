import os, pdb, pickle, sys, shutil, random
from re import sub
from tqdm import tqdm
import numpy as np

"""For organising all files in shallow directory into separate directories, based on their name"""

src_dir_path = sys.argv[1]
dest_parent_dir = sys.argv[2]
dest_dirs = [dest_parent_dir, dest_parent_dir +'/train', dest_parent_dir +'/val']
for this_dir in dest_dirs:
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
_,sub_dirs,_ = next(os.walk(src_dir_path))
src_subdir_paths = [os.path.join(src_dir_path, d) for d in sub_dirs]
split_ratio = 0.8
random.shuffle(src_subdir_paths)
train, val = src_subdir_paths[:int(len(src_subdir_paths)*split_ratio)], src_subdir_paths[int(len(src_subdir_paths)*split_ratio):]
# singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10','m11','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
# new_dirs = ['_male1_','_male2','_male3','_male4','_male5','_male6','_male7','_male8','_male9','_male10','_male11','female1','female2','female3','female4','female5','female6','female7','female8','female9']
# test_data = ['female8', 'female9', '_male11', '_male2', '_male8']

for i, src_subdir_paths in enumerate([train, val]):
    for src_subdir_path in src_subdir_paths:
        _,_, files = next(os.walk(src_subdir_path))
        for file in files:  
            if i == 0:
                dst_subdir = os.path.join(dest_dirs[1], os.path.basename(src_subdir_path))
                if not os.path.exists(dst_subdir):
                    os.mkdir(dst_subdir)
                shutil.move(os.path.join(src_subdir_path, file), os.path.join(dst_subdir, file))
            else:
                dst_subdir = os.path.join(dest_dirs[2], os.path.basename(src_subdir_path))
                if not os.path.exists(dst_subdir):
                    os.mkdir(dst_subdir)
                shutil.move(os.path.join(src_subdir_path, file), os.path.join(dst_subdir, file))


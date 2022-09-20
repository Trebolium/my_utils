import os, shutil, pdb, argparse
from tqdm import tqdm
from random import shuffle

"""Takes a directory with train and val sets, where each set has different speakers
Converts these into sets with same speakers, but different uttrs examples
"""

parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--src_dir', type=str, default='')
parser.add_argument('-dd', '--dst_dir', type=str, default='')
parser.add_argument('-r', '--split_ratio', type=float, default=0.8)
config = parser.parse_args()
total_per_voice = {}

file_path_list = []
existing_dirs = []
train_split = config.split_ratio
val_split = 1 - train_split

if not os.path.exists(config.dst_dir):
    os.makedirs(os.path.join(config.dst_dir, 'train'))
    os.makedirs(os.path.join(config.dst_dir, 'val'))
_, dirs, _ = next(os.walk(config.src_dir))

# go through tiers, collect all file_paths, making new speaker dirs in dst_dir as you go
for dir in dirs:
    _, subdirs, _ = next(os.walk(os.path.join(config.src_dir, dir)))
    for subdir in subdirs:
        os.mkdir(os.path.join(config.dst_dir, 'train', subdir))
        os.mkdir(os.path.join(config.dst_dir, 'val', subdir))
        _, _, files = next(os.walk(os.path.join(config.src_dir, dir, subdir)))
        total_per_voice[subdir] = len(files)
        for f in files:
            f_path = os.path.join(config.src_dir, dir, subdir, f)
            file_path_list.append(f_path)

shuffle(file_path_list)
per_voice_counter = total_per_voice.copy()
for key in per_voice_counter.keys(): per_voice_counter[key] = 0

for f_path in tqdm(file_path_list):
    this_dir = f_path.split('/')[-2] #get the dir this file was in
    # check if each subdir has reached its quota for train split
    if per_voice_counter[this_dir] <= train_split*total_per_voice[this_dir]: #send to train
        shutil.copy(f_path, os.path.join(config.dst_dir, 'train', this_dir))
        per_voice_counter[this_dir] += 1
    else: #send to test
        shutil.copy(f_path, os.path.join(config.dst_dir, 'val', this_dir))
        

    
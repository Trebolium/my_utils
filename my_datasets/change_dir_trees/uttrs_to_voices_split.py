import os, shutil, pdb, argparse, math
from random import shuffle

"""Takes a directory with train and val sets, where each set has different utterances from the same speakers
Converts these into sets where speakers are different - split by voice_id instead of utterances.
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
_, split_dirs, _ = next(os.walk(config.src_dir))

all_voice_dirs = []
# go through tiers, collect all file_paths, making new speaker dirs in dst_dir as you go
for split_dir in split_dirs:
    _, voice_dirs, _ = next(os.walk(os.path.join(config.src_dir, split_dir)))
    for voice_dir in voice_dirs:
        if voice_dir not in all_voice_dirs: # add to this list across all split_dirs
            all_voice_dirs.append(voice_dir)
        _, _, files = next(os.walk(os.path.join(config.src_dir, split_dir, voice_dir)))
        # register total ammount of uttrs per singer
        if voice_dir in total_per_voice:
            total_per_voice[voice_dir] += len(files)
        else:
            total_per_voice[voice_dir] = len(files)
        #populate file_path_list from files
        for f in files:
            f_path = os.path.join(config.src_dir, split_dir, voice_dir, f)
            file_path_list.append(f_path)

shuffle(file_path_list)
shuffle(all_voice_dirs)
train_voices = all_voice_dirs[:math.floor(len(all_voice_dirs)*train_split)]

for f_path in file_path_list:
    this_voice_dir = f_path.split('/')[-2] #get the dir this file was in
    # check if each subdir has reached its quota for train split
    if this_voice_dir in train_voices:
        new_path = os.path.join(config.dst_dir, 'train', this_voice_dir)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.copy(f_path, new_path)
        print(f'Copied {f_path} to {new_path}')

    else: #send to test
        new_path = os.path.join(config.dst_dir, 'val', this_voice_dir)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.copy(f_path, new_path)
        print(f'Copied {f_path} to {new_path}')


    
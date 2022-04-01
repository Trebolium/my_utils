import os, pdb, shutil, pickle, sys, librosa
from tdqm import tdqm
import numpy as np

"""
    Counts all audio files in a 1-tier directory
    collects audio durations and produces the dataset stats for:
        List of files that couldn't be computes
        List of durations
        Number of files
        Average duration
        Median duration

    Requires src and dst directories as arguments
"""

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
dir_dict = {}

_, dirs, _ = next(os.walk(src_dir))

length_list = []
error_list = []

for i, dir in tdqm(enumerate(dirs)):
    _,_, files = next(os.walk(dir))
    num_files = len(files)
    dir_dict[dir] = num_files
    for j, f in enumerate(files):
        if f.endswith('.m4a') and not f.startswith('.'):
            # if num == 432 or num == 821:
            #     pdb.set_trace()
            try:
                wav, sr = librosa.load(os.path.join(src_dir, dir, f), sr=None)
                wav_dur = len(wav) / sr
                print(f, wav_dur)
                length_list.append(wav_dur)
            except:
                error_list.append(os.path.join(src_dir, dir, f))

length_arr = np.asarray(length_list)

with open(os.path.join(dst_dir, 'dataset_durations.pkl'), 'wb') as handle:
    pickle.dump({'error_list':error_list, 'length_list':length_list, 'num_files':len(length_arr), 'average_dur':np.average(length_arr), 'mediam_dur':np.median(length_arr)}, handle)

print(f'Average is {np.average(length_arr)}')
sorted_dict = sorted(dir_dict.items(), key=lambda item: item[1])
print('Play with dictionary \'sorted_dict\' to get an idea of the distribution of directory sizes (directories names are 8 digit long)'.upper())

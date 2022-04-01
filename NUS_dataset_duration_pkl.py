import os, pdb, shutil, pickle, librosa, argparse
import numpy as np

"""
    Goes through files in a directory
    Uses directory tree (dir_tree) to navigate through the dataset
    collects duration data
"""

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src_dir", type=str, default='/Volumes/LaCie/datasets/VCTK/wav48_silence_trimmed')
parser.add_argument("-e", "--file_ext", type=str, default='mp3')
config = parser.parse_args()

src_dir = config.src_dir
file_ext = config.file_ext

_, dirs, _ = next(os.walk(src_dir))
error_file_list = []
per_singer_info = []
for i, dir in enumerate(dirs):
    sum_dur_per_singer = 0
    _,_, files = next(os.walk(os.path.join(src_dir,dir)))
    num_files = len(files)
    for j, f in enumerate(files):
        if f.endswith(file_ext) and not f.startswith('.'):
            if f[-6] == '1':
                try:
                    wav, sr = librosa.load(os.path.join(src_dir, dir, f), sr=None)
                    wav_dur = len(wav) / sr
                    sum_dur_per_singer += wav_dur 
                except:
                    error_file_list.append(os.path.join(src_dir, dir, f))
    singer_data_dict = {'singer_name':dir, 'num_files':num_files, 'summed_duration':sum_dur_per_singer}
    print(round(i*100/len(dirs)), singer_data_dict)
    per_singer_info.append(singer_data_dict) 

avg_singer_duration = np.average(np.asarray([data['summed_duration'] for data in per_singer_info]))
std_singer_duration = np.std(np.asarray([data['summed_duration'] for data in per_singer_info]))
print(f'avg_singer_duration: {avg_singer_duration}, std_singer_duration: {std_singer_duration}')
dataset_dict = {'avg_singer_duration':avg_singer_duration, 'std_singer_duration':std_singer_duration, 'per_singer_dict_list': per_singer_info, 'error_file_list':error_file_list}

with open(os.path.join(src_dir, 'dataset_duration_data.pkl'), 'wb') as handle:
    pickle.dump(dataset_dict, handle)
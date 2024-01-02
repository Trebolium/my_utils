import os, pdb, shutil, pickle, librosa, argparse, yaml, sys
import numpy as np
if os.path.abspath('/homes/bdoc3/my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('/homes/bdoc3/my_utils'))
from my_os import recursive_file_retrieval
from tqdm import tqdm

"""
    Goes through fps in a directory
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

all_dirs, fps = recursive_file_retrieval(src_dir)
fps = [fp for fp in fps if fp.endswith(config.file_ext)]
voice_dirs = all_dirs[3:]
voice_names = [os.path.basename(d) for d in voice_dirs]
summed_durations = 0
num_files = len(fps)

for fp in tqdm(fps):
    if fp.endswith(file_ext) and not fp.startswith('.'):
        if config.file_ext == 'npy':

            with open(
                os.path.join(config.src_dir, "feat_params.yaml")
            ) as File:
                feat_params = yaml.load(File, Loader=yaml.FullLoader)
            # pdb.set_trace()
            feats = np.load(fp)
            num_ts = feats.shape[0]
            summed_durations += num_ts * (feat_params['frame_dur_ms']/1000)

        else:
            voice_dir = os.path.basename(os.path.dirname(fp))
            wav, sr = librosa.load(os.path.join(src_dir, voice_dir, fp), sr=None)
            wav_dur = len(wav) / sr
            summed_durations += wav_dur
         


    singer_data_dict = {'singer_name':voice_names, 'num_files':num_files, 'summed_duration':summed_durations}
    # print(round(i*100/len(dirs)), singer_data_dict)
    # per_singer_info.append(singer_data_dict) 

# avg_singer_duration = np.average(np.asarray([data['summed_duration'] for data in per_singer_info]))
# std_singer_duration = np.std(np.asarray([data['summed_duration'] for data in per_singer_info]))
# print(fp'avg_singer_duration: {avg_singer_duration}, std_singer_duration: {std_singer_duration}')
# dataset_dict = {'avg_singer_duration':avg_singer_duration, 'std_singer_duration':std_singer_duration, 'per_singer_dict_list': per_singer_info, 'error_file_list':error_file_list}
print('Average duration:', summed_durations/len(voice_names))
pdb.set_trace()
with open(os.path.join(src_dir, 'dataset_duration_data.pkl'), 'wb') as handle:
    pickle.dump(dataset_dict, handle)
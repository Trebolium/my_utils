from audio_feats import add_butter_noise, audio_to_mel_autovc, db_normalize
from utils import split_subsets_by_subdirs, get_dst_file_path, recursive_file_retrieval, make_dataset_dir
import pdb, librosa, os
from librosa.filters import mel
import numpy as np
from tqdm import tqdm

"""DAMP DATASET"""
src_dir = '/import/c4dm-datasets/DAMP_Intonation_Dataset/vocal_tracks'
dst_dir = '/homes/bdoc3/my_data/spmel_data/damp_inton/unnormmed_mels_split_by_voices'

# """VCTK DATASET"""
# src_dir = '/import/c4dm-datasets/VCTK-Corpus-0.92/wav48_silence_trimmed'
# dst_dir = '/homes/bdoc3/my_data/spmel_data/vctk/unnormmed_mels_split_by_voices'

train_split = 0.8
sr = 16000
hop_size = 256
frame_dur_ms = hop_size / sr * 1000
feat_params = {'type':'mels', "fmin":50, "fmax":7600, 'num_feats':80, 'frame_dur_ms':frame_dur_ms, 'sr':sr, 'fft_size':1024, 'hop_size':hop_size} # pw.default_frame_dur_ms is 5ms. We've changed the frame_dur to 10 ms  
mel_filter = mel(feat_params['sr'], feat_params['fft_size'], fmin=feat_params['fmin'], fmax=feat_params['fmax'], n_mels=feat_params['num_feats']).T
min_level = np.exp(-100 / 20 * np.log(10))
overwrite_dst_files = False

# collect paths and arrange by subsets
dir_list, audio_paths = recursive_file_retrieval(src_dir)

# """VCTK DATASET"""
# filtered_paths = [path for path in audio_paths if path[-9:-5]=='mic1' and path.endswith('flac')]
"""DAMP DATASET"""
filtered_paths = [path for path in audio_paths if path.endswith('m4a')]

subset_list = split_subsets_by_subdirs(filtered_paths, train_split)
subset_names = ['train', 'val']

make_dataset_dir(dst_dir, feat_params)

exceptions_list = []
for i, subset in enumerate(subset_list):
    print(f'starting {subset_names[i]} set...')
    for file_path in tqdm(subset):
        dst_file_path = get_dst_file_path(file_path, dst_dir, i)
        if os.path.exists(dst_file_path) and overwrite_dst_files == False:
            continue
        audio, _ = librosa.load(file_path, sr=feat_params['sr'])
        try:
            btr_noiz_audio = add_butter_noise(audio, feat_params['sr'])
            melspec = audio_to_mel_autovc(btr_noiz_audio, feat_params['fft_size'], feat_params['hop_size'], mel_filter)
            unnormed_db_melspec = db_normalize(melspec, min_level, normalise=False)
            np.save(dst_file_path, unnormed_db_melspec)
        except Exception as e:
            log_string = f'Skipping {os.path.basename(file_path)} as it caused the following error: {e}'
            print(f'Skipping {os.path.basename(file_path)} as it caused the following error: {e}')
            exceptions_list.append(log_string)
            continue
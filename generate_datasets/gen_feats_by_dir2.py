import sys, os, librosa, yaml, pdb
import numpy as np
from tqdm import tqdm
sys.path.insert(1, '/homes/bdoc3/my_utils')
from schluterNorm_dataset import normalise_data
from my_os import recursive_file_retrieval
from audio.worldvocoder import generate_params, chandna_feats, gen_world_feat
from audio.editor import trim_silence_splice
from audio.mel import audio_to_mel_librosa

if __name__ == '__main__':

    # dirs
    copied_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    f_ext = '.npy'
    dataset_path = '/import/c4dm-datasets/DAMP_Intonation_Dataset/vocal_tracks'
    del_unnormed = False
    normalise = True

    # params
    frame_dur_ms = 5.80498866 # represents the typical duration for the common ratio of 256/44100
    frame_dur_s = frame_dur_ms/100
    sr = 16000
    fft_size = 1024
    fmin=50
    fmax=1100
    num_feats=80
    feat_type='mel'
    hop_size = round(sr * frame_dur_s)
    # feat_params = generate_params(False, sampling_rate, frame_dur_ms)
    feat_params = {'type':'harmsOnly', "fmin":fmin, "fmax":fmax, 'num_feats':num_feats, 'frame_dur_ms':frame_dur_ms, 'sr':sr, 'fft_size':fft_size, 'hop_size':hop_size} # pw.default_frame_dur_ms is 5ms. We've changed the frame_dur to 10 ms  


    # gather files from copied_dir and dataset
    _, copied_dir_all_fps = recursive_file_retrieval(copied_dir)
    copied_dir_all_fs = [os.path.basename(path) for path in copied_dir_all_fps if path.endswith('.npy')]
    _, dataset_path_all_fps = recursive_file_retrieval(dataset_path)

    pdb.set_trace()

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    with open(os.path.join(dst_dir, 'feat_params.yaml'), 'w') as File:
        yaml.dump(feat_params, File, default_flow_style=False)


    counter = 0
    exceptions = []
    for copied_dir_fps in tqdm(copied_dir_all_fps):
        if copied_dir_fps.endswith(f_ext):
            # check / setup relevant dirs/paths
            fn = os.path.basename(copied_dir_fps)
            voice_dir = copied_dir_fps.split('/')[-2]
            subset = copied_dir_fps.split('/')[-3]
            file_dst_path = os.path.join(dst_dir, subset, voice_dir, fn)
            if os.path.exists(file_dst_path):
                continue
            if not os.path.exists(os.path.join(dst_dir, subset, voice_dir)):
                os.makedirs(os.path.join(dst_dir, subset, voice_dir))

            # process audio    
            try:
                formatted_audio = trim_silence_splice(os.path.join(dataset_path, fn[:-4]+'.m4a'), feat_params)
                feats = audio_to_mel_librosa(formatted_audio, sr, fft_size, hop_size, num_feats, fmin, fmax)
                # feats = gen_world_feat(formatted_audio.astype('double'), feat_params)
            except Exception as e:
                print(f'Skipping file {fn} as the following exception occured: {e}')
                exceptions.append(e)
                continue

            # save features
            np.save(file_dst_path, feats)

            # counter += 1
            # if counter >= total_files_allowed:
            #     print(f'Dst directory now has {total_files_allowed} files. Breaking loop')
            #     break
    
    # normalise data
    if normalise:
        normalise_data(dst_dir, os.path.join(dst_dir, dst_dir + '_schluterNormmed'), del_unnormed)

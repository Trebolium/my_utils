import sys, os, librosa, yaml, pdb
import soundfile as sf
import numpy as np
from tqdm import tqdm

# for some reason there is an unwatend path in sys.path. Must figure out how to remove this
for i in sys.path:
    if i == '/homes/bdoc3/wavenet_vocoder':
        sys.path.remove(i)

sys.path.insert(1, '/homes/bdoc3/my_utils')
# from schluterNorm_dataset import normalise_data
from my_os import recursive_file_retrieval
from audio.world import chandna_feats

"""This script collects all files from a src datasets, and copies files to a dst dataset that does not yet have them.
    Check to ensure that the ds your are copying from has the same tree structure as the desired dst directory
    """


if __name__ == '__main__':

    # dirs
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    f_ext = '.m4a'
    # dataset_path = '/import/c4dm-datasets/DAMP_Intonation_Dataset/vocal_tracks'

    # params
    # # frame_dur_ms = 5.80498866 # represents the typical duration for the common ratio of 256/44100
    # frame_dur_ms = 5
    # frame_dur_s = frame_dur_ms/100
    # sr = 16000
    # fft_size = 1024
    # fmin=50
    # fmax=1100
    # num_feats=80
    # feat_type='mel'
    # hop_size = round(sr * frame_dur_s)
    # # feat_params = generate_params(False, sampling_rate, frame_dur_ms)
    # feat_params = {'type':'harmsOnly',
    #                 'fmin':fmin,
    #                 'fmax':fmax,
    #                 'num_feats':num_feats,
    #                 'frame_dur_ms':frame_dur_ms,
    #                 'sr':sr,
    #                 'fft_size':fft_size,
    #                 'hop_size':hop_size} # pw.default_frame_dur_ms is 5ms. We've changed the frame_dur to 10 ms  


    # gather files from src_dir and dataset
    _, src_dir_all_fps = recursive_file_retrieval(src_dir)
    _, dst_dir_all_fps = recursive_file_retrieval(dst_dir)

    # if not os.path.exists(dst_dir):
    #     os.mkdir(dst_dir)
    # with open(os.path.join(dst_dir, 'feat_params.yaml'), 'w') as File:
    #     yaml.dump(feat_params, File, default_flow_style=False)


    counter = 0
    exceptions = []
    missing_fns = []
    missing_voices = []

    for src_dir_fps in tqdm(src_dir_all_fps):
        if src_dir_fps.endswith(f_ext):
            fn = os.path.basename(src_dir_fps)
            voice_id = fn.split('_')[0]
            fn_missing = True
            voice_id_missing = True
            for fp in dst_dir_all_fps:
                if fn in fp:
                    fn_missing = False
                if voice_id in fp:
                    voice_id_missing = False
            if fn_missing:
                missing_fns.append(fn)
            if voice_id_missing:
                missing_voices.append(voice_id)

    with open('missing_filenames.txt', 'w') as f:
        for item in missing_fns:
            print(f'saving {item} to missing_filenames.txt')
            f.write("%s\n" % item)

    with open('missing_voices.txt', 'w') as f:
        for item in missing_voices:
            print(f'saving {item} to missing_voices.txt')
            f.write("%s\n" % item)

    pdb.set_trace()
            # # CONFIGURE SPECIFIC TO DATASET (CURRENTLY SET FOR DAMP INTONATION DS)
            # fn = os.path.basename(src_dir_fps)
            # subset = src_dir_fps.split('/')[-3]
            # if src_uses_subsubdirs:
            #     subsubset = src_dir_fps.split('/')[-2]
            # else:
            #     subsubset = fn.split('_')[0]
            # file_dst_path = os.path.join(dst_dir, subset, subsubset, fn)
            
            # if os.path.exists(file_dst_path):
            #     continue
            # if not os.path.exists(os.path.join(dst_dir, subset, subsubset)):
            #     os.makedirs(os.path.join(dst_dir, subset, subsubset))

            # # process audio
            # try:
            #     # formatted_audio = trim_silence_splice(os.path.join(dataset_path, fn[:-4]+'.m4a'), feat_params)
            #     # feats = audio_to_mel_librosa(formatted_audio, sr, fft_size, hop_size, num_feats, fmin, fmax)
            #     # feats = gen_world_feat(formatted_audio.astype('double'), feat_params)
            #     y, _ = sf.read()
            #     feats = chandna_feats(src_dir_fps, feat_params)
            # except Exception as e:
            #     print(f'Skipping file {fn} as the following exception occured: {e}')
            #     exceptions.append(e)
            #     continue

            # # save features
            # np.save(file_dst_path, feats)

            # counter += 1
            # if counter >= total_files_allowed:
            #     print(f'Dst directory now has {total_files_allowed} files. Breaking loop')
            #     break
    
    # normalise data
    # if normalise:
    #     normalise_data(dst_dir, os.path.join(dst_dir, dst_dir + '_schluterNormmed'), del_unnormed)

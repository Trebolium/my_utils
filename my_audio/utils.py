import os
import pdb
import warnings

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import crepe

from my_audio.world import get_world_feats
from my_audio.editor import desilence_concat_audio
from my_audio.mel import audio_to_mel_autovc, db_normalize, add_butter_noise


"""
Function designed to convert any dataset into a given set of features (usually mel spectrograms, sometimes World vocoder features)
Intended to be used to create dataset directories. Therefore includes:
    feat_params.yaml file generation
    creationg of a dir with subsets train, val, test
    
"""

def audio_io(file_path, trg_sr):
    with warnings.catch_warnings(): # warning 
        warnings.simplefilter("ignore", category=UserWarning)
    file_path = str(file_path)
    if file_path.endswith('.m4a'):
        y, _ = librosa.load(file_path, sr=trg_sr)
    else:
        y, samplerate = sf.read(file_path)
        if samplerate != trg_sr:
            if not y.flags['F_CONTIGUOUS']:
                y = librosa.resample(np.asfortranarray(y), orig_sr=samplerate, target_sr=trg_sr)
            else:
                y = librosa.resample(y, orig_sr=samplerate, target_sr=trg_sr)
    return y

def audio2feats_process(iterables_list):
    file_path = iterables_list[0]
    config = iterables_list[1] # all entries after 0 currently come from the args variable in main script
    feat_params = iterables_list[2]

    #FIXME: Should also include test set
    #FIXME: This block should be executed outside this processing function
    # generate destination path
    if 'train' in file_path:
        subset = 'train'
    elif 'val' in file_path:
        subset = 'val'
    elif 'test' in file_path:
        subset = 'test'
    else:
        subset = ''
    
    if len(os.path.basename(file_path).split('_')) > 1:
        fp_dir = os.path.join(config.dst_dir, subset, os.path.basename(file_path).split('_')[0])
    elif len(os.path.basename(file_path).split('-')) > 1:
        fp_dir = os.path.join(config.dst_dir, subset, os.path.basename(file_path).split('-')[0])
    else:
        raise NotImplementedError
    
    if not os.path.exists(fp_dir):
        os.makedirs(fp_dir)

    # generate destination path and check to see is it available
    dst_file_path = os.path.join(fp_dir, os.path.basename(file_path)[:-len(config.audio_ext)] +config.numpy_ext)
    if os.path.exists(dst_file_path):
        return print(f'path {dst_file_path} exists. Skipping.')

    # if file is m4a, pydub can read it
    y = audio_io(file_path, feat_params['sr'])

    if len(y.shape) == 2:
        if config.channel_choice == 'left':
            y = y[:,0]
        else:
            y = y[:,1]

    print(f'processing {file_path}')

    if config.desilence:
        y = desilence_concat_audio(y, feat_params['sr'])

    if config.feat_type == 'world':
        feats = get_world_feats(y, feat_params)
    elif config.feat_type == 'mel':
        y = add_butter_noise(y, feat_params['sr'])
        mel_filter = iterables_list[3]
        min_level = iterables_list[4]
        hop_size = iterables_list[5]
        db_unnormed_melspec = audio_to_mel_autovc(y, feat_params['fft_size'], hop_size, mel_filter)
        feats = db_normalize(db_unnormed_melspec, min_level)
    elif config.feat_type == 'crepe':
        _, frequency_prediction, confidence, _ = crepe.predict(y, feat_params['sr'], viterbi=False, step_size=feat_params['frame_dur_ms'])

    if config.feat_type == 'crepe':
        np.savez(dst_file_path, frequency_prediction, confidence)
    else:
        feats = feats.astype(np.float32)
        np.save(dst_file_path, feats)
    # return print(f'finished {file_path}')






import numpy as np
import sys, os, pdb, argparse, datetime, time, pickle, librosa
from tqdm import tqdm
import concurrent.futures
from pydub import AudioSegment
import soundfile as sf
from librosa.filters import mel
sys.path.insert(1, '/homes/bdoc3/my_utils')

from my_os import recursive_file_retrieval
from my_container import substring_exclusion, substring_inclusion, balance_by_strings, separate_by_starting_substring
from generate_datasets.utils import make_dataset_dir
from my_threads import multithread_chunks
from my_audio.world import get_world_feats
from my_audio.mel import audio_to_mel_autovc, db_normalize

"""Requires a pickled list. Goes through entries, collects filenames and their parent dirs
    Appends these to the path for the original dataset they came from
    and re-generates the numpys"""

"""This function designed to work with multithreading"""
def edit_convert_save_track(iterables_list):
    file_path = iterables_list[0]
    config = iterables_list[1]
    feat_params = iterables_list[2]

    # generate destination path
    if 'train' in file_path:
        subset = 'train'
        fp_dir = os.path.join(config.dst_dir, subset, os.path.basename(file_path).split('_')[0])
    elif 'val' in file_path:
        subset = 'val'
        fp_dir = os.path.join(config.dst_dir, subset, os.path.basename(file_path).split('_')[0])
    else:
        fp_dir = os.path.join(config.dst_dir, os.path.basename(file_path).split('_')[0])
    # CHECK THAT FILENAMES ARE IN COMPATIBLE FORMAT FOR THIS
    if not os.path.exists(fp_dir):
        os.makedirs(fp_dir)

    # generate destination path and check to see is it available
    dst_file_path = os.path.join(fp_dir, os.path.basename(file_path)[:-(len(config.ext)+1)] +'.npy')
    # pdb.set_trace()
    # if os.path.exists(dst_file_path):
    #     return print(f'path {file_path} exists. Skipping.')

    # if file is m4a, only pydub can read it
    try:
        if file_path.endswith('m4a'):
            y, samplerate = librosa.load(file_path, sr=feat_params['sr'])
            # spit it out as wav audio and read in as 
            # y.export("tmp.wav", format="wav")
            # y, samplerate = sf.read("tmp.wav") 
        else:
            y, samplerate = sf.read(file_path)
        
    except Exception as e:
    # except (OSError, IndexError) as e:
        return print(f'Exception: {e} caused by: {file_path}')
        
    
    if samplerate != feat_params['sr']:
        y = librosa.resample(y,samplerate, feat_params['sr'])

    try:
        if config.feat_type == 'world':
            feats = get_world_feats(y, feat_params)
        elif config.feat_type == 'mel':
            mel_filter = iterables_list[3]
            min_level = iterables_list[4]
            hop_size = iterables_list[5]
            db_unnormed_melspec = audio_to_mel_autovc(y, feat_params['fft_size'], hop_size, mel_filter)
            feats = db_normalize(db_unnormed_melspec, min_level)
        feats = feats.astype(np.float32)
        np.save(dst_file_path, feats)
        return print(f'finished {file_path}')

    except Exception as e:
    # except IndexError as e:
        return print(f'Exception: {e} caused by {file_path}')

parser = argparse.ArgumentParser(description='params for converting audio to spectral using world', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# script variables
parser.add_argument('-upl','--use_path_list', type=str, default='') 
parser.add_argument('-dd','--dst_dir', default='/import/c4dm-02/bdoc3/spmel/damp_inton_80', type=str)
parser.add_argument('-sd','--src_dir', default='/homes/bdoc3/my_data/audio_data/damp_desilenced_concat', type=str)
parser.add_argument('-p','--pkl', default='', type=str)
parser.add_argument('-e','--ext', default='m4a', type=str)
parser.add_argument('-np','--num_processes', default=16, type=int)    
#feat params (bool, str, int)
parser.add_argument('-ft','--feat_type', default='mel', type=str)
parser.add_argument('-wp','--w2w_process', default='wav2world', type=str)
parser.add_argument('-drm','--dim_red_method', default='chandna', type=str)
parser.add_argument('-fdm','--frame_dur_ms', default=5, type=int)    
parser.add_argument('-nhf','--num_harm_feats', default=80, type=int)
parser.add_argument('-naf','--num_aper_feats', default=4, type=int)
parser.add_argument('-sr','--sampling_rate', default=16000, type=int)
parser.add_argument('-fs','--fft_size', default=None, type=int)
parser.add_argument('-fmin', default=50, type=int) #50 chosen by me, 71 chosen by default params   
parser.add_argument('-fmax', default=1100, type=int) #1100 chosen by me, 800 chosen by default params 
parser.add_argument("-n", "--notes", type=str, default='', help= "Add these notes which will be saved to a config text file that gets saved in your saved directory")

config = parser.parse_args()

# declaring variables
counter = 0
train_split = 0.8
subset_names = ['train', 'val']
original_audio_path_list = []
feat_params = {'w2w_process':config.w2w_process,                           
                'dim_red_method':config.dim_red_method,
                'num_harm_feats':config.num_harm_feats,
                'num_aper_feats':config.num_aper_feats,
                'frame_dur_ms':config.frame_dur_ms,
                'sr':config.sampling_rate,
                'fft_size':config.fft_size,
                "fmin":config.fmin,
                "fmax":config.fmax
                }

exceptions = pickle.load(open(config.pkl, 'rb'))
# keep fn and its parent dir, and append this to the src_dir
filtered_list = [os.path.join(config.src_dir, os.path.basename(path).split('_')[0], os.path.basename(path)[:-3]+config.ext) for types, path in exceptions if path.endswith('npy')]

if config.feat_type == 'mel':
    mel_filter = mel(feat_params['sr'], feat_params['fft_size'], fmin=feat_params['fmin'], fmax=feat_params['fmax'], n_mels=feat_params['num_harm_feats']).T
    # self.mel_filter = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    hop_size = int((feat_params['frame_dur_ms']/1000) * feat_params['sr'])
    args = [config, feat_params, mel_filter, min_level, hop_size]
elif config.feat_type == 'world':
    args = [config, feat_params]
else:
    raise Exception('feat_type param was not recognised.')

multithread_chunks(edit_convert_save_track, filtered_list, config.num_processes, args)

# for fp in filtered_list:
#     arg_list = [fp] + args
#     try:
#         pdb.set_trace()
#         edit_convert_save_track(arg_list)
#     except Exception as e:
#         print(e)
#         pdb.set_trace()
#         edit_convert_save_track(arg_list)

# for i in tqdm(range(0, len(exceptions), config.num_processes)):
#     et = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
#     print(f'Elapsed: {et}: Progress {i}/{len(exceptions)}')
#     fps_chunk_list = exceptions[i:i + config.num_processes]
#     # necessary to ensure for that lists reflect size of last chunk
#     config_list = [config for i in range(len(fps_chunk_list))]
#     feat_p_list = [feat_params for i in range(len(fps_chunk_list))]
#     iterables_list = list(zip(fps_chunk_list, config_list, feat_p_list))
#     # pdb.set_trace()
#     # edit_convert_save_track(iterables_list[0])
#     # https://www.youtube.com/watch?v=fKl2JW_qrso, https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/MultiProcessing/multiprocessing-demo.py
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = executor.map(edit_convert_save_track, iterables_list)

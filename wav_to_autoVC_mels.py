import os, pdb, pickle, yaml, time, sys, argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm
from my_utils import recursive_file_retrieval, substring_exclusion, substring_inclusion, balance_class_list, separate_into_groups
from audio import butter_highpass, pySTFT, audio_to_mel
from librosa.filters import mel

def str2bool(v):
    return v.lower() in ('true')

def path_list_by_rules(rootDir, exclude_list, class_list):
    fileList = recursive_file_retrieval(rootDir)
    classes_only_list = substring_exclusion(fileList, exclude_list) 
    classes_only_list = substring_inclusion(classes_only_list, class_list)
    classes_only_list  = [f for f in classes_only_list if f[-6] == '_'] # to ensure that the format of the file ends with '_' and a vowel
    files_by_singers = separate_into_groups(classes_only_list, singer_list)
    total_balanced_list = []
    for f_by_singer in files_by_singers:
        total_balanced_list.extend(balance_class_list(f_by_singer, class_list, 10))
    return total_balanced_list 

if __name__ == '__main__':

    #define audio directory
    singer_list = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10','m11','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
    class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
    exclude_list = ['caro','row','long','dona']
    original_audio_path_list = []
    desired_sir = 16000
    frame_size = 400
    hop_size = 160
    n_mels = 40
    fmin = 90
    fmax = 7600

    mel_filter = mel(desired_sir, frame_size, fmin=fmin, fmax=fmax, n_mels=n_mels).T
    w_param = {"sr":desired_sir, "fmin":90, "fmax":7600, 'num_feats':40, 'frame_size':400, 'hop_size':160}

    parser = argparse.ArgumentParser(description='params for converting audio to spectral using world', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n','--use_npss', type=str2bool, default=False)
    parser.add_argument('-o','--overwrite_dst_files', type=str2bool, default=False) 
    parser.add_argument('-s','--src_dir', type=str, default='/import/c4dm-datasets/VocalSet1-2/data_by_singer')
    parser.add_argument('-d','--dst_dir', type=str, default='/homes/bdoc3/my_data/world_vocoder_data/some_dir')
    config = parser.parse_args()
    dst_dir = config.dst_dir
    use_npss = config.use_npss

    original_audio_path_list = path_list_by_rules(config.src_dir, exclude_list, class_list)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    file = open(os.path.join(dst_dir, 'w_params.yaml'), 'w')
    yaml.dump(w_param, file)
    file.close()

    pdb.set_trace()

    for file_path in tqdm(original_audio_path_list):
        # check if converted file already exists
        dst_path = dst_dir +'/' +os.path.basename(file_path)[:-4] +'.npy'
        if os.path.exists(dst_path):
            if config.overwrite_dst_files == False:
                print(f'{dst_path} already exists. Skipping')
                continue

        # processing
        audio, current_sr = sf.read(file_path)
        melspec = audio_to_mel(audio, current_sr, desired_sir, w_param['frame_size'], w_param['hop_size'], mel_filter)

        # handle numpy
        if np.isnan(melspec).any() == True:
            print(f'nan found is comp_sp for file: {file_path}')
            pdb.set_trace() 
        np.save(dst_path, melspec)
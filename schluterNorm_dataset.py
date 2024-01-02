from copy import Error
import numpy as np
import sys, os, torch, shutil, pdb
from tqdm import tqdm
if os.path.abspath('.../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('.../my_utils'))
from my_os import recursive_file_retrieval
from my_audio.utils import audio_io
from my_normalise import TorchStatsRecorder, get_norm_stats, apply_norm_stats


if __name__ == '__main__':
    sd = sys.argv[1]
    dd = sys.argv[2]
    sr = sys.argv[3]
    ext = sys.argv[4]
    if not os.path.exists(dd):
        os.makedirs(dd)
    total_mean, total_std, total_errs = get_norm_stats(os.path.join(sd, 'train'))
    dirs, fps = recursive_file_retrieval(sd, return_parent=False)
    normed_feats = []
    non_normed_feats = []
    for fp in tqdm(fps):
        
        fn = os.path.basename(fp)
        if not fn.endswith(ext):
            if fn.endswith('yaml'):
                shutil.copy(fp, os.path.join(dd, os.path.basename(fp)))
            continue
        
        if ext == 'npy':
            feats = np.load(fp)
        elif ext == 'wav':
            feats = audio_io(fp, sr)

        normed_feats = apply_norm_stats(feats, total_mean, total_std)
        f_rel_path = os.path.relpath(fp, sd)
        dfp = os.path.join(dd, f_rel_path)
        df_dir = os.path.dirname(dfp)
        if not os.path.exists(df_dir):
            os.makedirs(df_dir)
        np.save(dfp, normed_feats) 
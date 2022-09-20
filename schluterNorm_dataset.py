from copy import Error
import numpy as np
import sys, os, torch, shutil, pdb
from tqdm import tqdm
sys.path.insert(1, '\homes\bdoc3\my_utils')
from my_os import recursive_file_retrieval
from my_normalise import TorchStatsRecorder


if __name__ == '__main__':
    pdb.set_trace()
    direc = sys.argv[1]
    total_mean, total_std, total_errs = get_norm_stats(direc)
    _, fps = recursive_file_retrieval(direc)
    normed_feats = []
    non_normed_feats = []
    for fp in fps:
        feats = np.load(fp)
        non_normed_feats.append(feats)
        normed_feats.append(apply_norm_stats(feats, total_mean, total_std))


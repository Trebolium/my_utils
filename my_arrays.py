import numpy as np
import random

def fix_feat_length(feats, crop_size, offset=None):
    random.seed(1)
    if feats.shape[0] > crop_size:
        diff = feats.shape[0] - crop_size
        if offset == None:
            offset = random.randint(0, diff)
        feats = feats[offset:offset+crop_size]
    else:
        diff = crop_size - feats.shape[0]
        offset = None
        if diff % 2 == 0:
            feats = np.pad(feats, pad_width=((diff//2, diff//2),(0,0)), mode='constant', constant_values=0)
        else:
            feats = np.pad(feats, pad_width=((diff//2, diff//2+1),(0,0)), mode='constant', constant_values=0)
    return feats, offset
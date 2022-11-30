import numpy as np
import random
import math
import torch
import warnings


"""
    Crop features or pad, depending on proposed crop size and feature length
    Return the negative value of the diff if crop_size is bigger than feats
    WARNING: If offset is negative it will be ignored and treated as if it were None
"""
def fix_feat_length(feats, crop_size, offset=None, this_seed=None):

    # if type(feats) != numpy.ndarray:
    #     warnings.warn('No array given to fix_feat_length - Returning NoneType.')
    #     return None

    if this_seed != None:
        random.seed(this_seed)
    diff = feats.shape[0] - crop_size
    # if feats longer than crop, choose random start point
    if feats.shape[0] > crop_size:
        if offset == None or offset < 0:
            offset = random.randint(0, diff)
        feats = feats[offset:offset+crop_size]

    # if feats shorter, pad on each side
    else:
        diff = crop_size - feats.shape[0]
        offset = -diff
        if feats.ndim == 1:
            if diff % 2 == 0:
                #pad_width determines length of padding for each axis as (before_axisN, after_axisN) 0 is appropriate for all feature min representations
                feats = np.pad(feats, pad_width=(diff//2, diff//2), mode='constant', constant_values=0)
            else:
                feats = np.pad(feats, pad_width=(diff//2, diff//2+1), mode='constant', constant_values=0)
        
        elif feats.ndim == 2:
            if diff % 2 == 0:
                #pad_width determines length of padding for each axis as (before_axisN, after_axisN) 0 is appropriate for all feature min representations
                feats = np.pad(feats, pad_width=((diff//2, diff//2),(0,0)), mode='constant', constant_values=0)
            else:
                feats = np.pad(feats, pad_width=((diff//2, diff//2+1),(0,0)), mode='constant', constant_values=0)

    return feats, offset


def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad



def container_to_tensor(container, add_batch_dim=False, device='cpu'):
    
    if type(container) == list:
        container = np.asarray(container)
    if type(container) == np.ndarray:
        container = torch.from_numpy(container).float()

    if add_batch_dim:
        container = container.unsqueeze(0)
    container = container.to(device)
    return container


def tensor_to_array(tensor):
    squeezed_tensor = torch.squeeze(tensor)
    arr = squeezed_tensor.detach().cpu().numpy()
    return arr


# find_runs taken with citation from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
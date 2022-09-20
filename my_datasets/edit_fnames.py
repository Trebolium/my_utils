from tqdm import tqdm
import os, sys, pdb

"""Alter the names of everything in a directory based on current strings"""

sys.path.insert(1, '/homes/bdoc3/my_utils')

from my_os import recursive_file_retrieval

_, all_fps = recursive_file_retrieval('/homes/bdoc3/my_data/world_vocoder_data/vctk')
all_fps = [p for p in all_fps if p.endswith('npy')]

pdb.set_trace()
for i, p in tqdm(enumerate(all_fps)):
    print(i)
    if p.endswith('npy.npy'):
        pdb.set_trace()
        new_p = p[:-7] +'.npy'
        os.rename(p, new_p)

import os, pdb, pickle, sys
from tqdm import tqdm
import numpy as np

"""For organising all files in shallow directory into separate directories, based on their name"""

src_dir_path = sys.argv[1]
singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10','m11','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
new_dirs = ['_male1_','_male2','_male3','_male4','_male5','_male6','_male7','_male8','_male9','_male10','_male11','female1','female2','female3','female4','female5','female6','female7','female8','female9']
test_data = ['female8', 'female9', '_male11', '_male2', '_male8']
subdirs = ['test', 'train']
dest_parent_dir = sys.argv[2]

for spkr_id in new_dirs:
    if spkr_id in test_data: spkr_id_path = os.path.join(dest_parent_dir, subdirs[0], spkr_id)
    else: spkr_id_path = os.path.join(dest_parent_dir, subdirs[1], spkr_id) 
    if not os.path.exists(spkr_id_path):
        os.makedirs(spkr_id_path) 
r, _, files = next(os.walk(src_dir_path))
for f in tqdm(files):
    if f.startswith('.') or f.endswith('.yaml'):
        continue
    for s_idx, singer in enumerate(singer_names):
        if singer in f:
            spec_feats = np.load(os.path.join(r, f))
            if new_dirs[s_idx] in test_data:
                np.save(os.path.join(dest_parent_dir, subdirs[0], new_dirs[s_idx], f[:-4]), spec_feats)
                break
            else:
                try:
                    np.save(os.path.join(dest_parent_dir, subdirs[1], new_dirs[s_idx], f[:-4]), spec_feats)
                    break
                except: pdb.set_trace()

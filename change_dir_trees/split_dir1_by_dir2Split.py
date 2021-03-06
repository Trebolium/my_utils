import sys, os, pdb
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval

# dataset directories
src_ds = sys.argv[1]
subject_ds = sys.argv[2]

# setup dir and tree
subsets = ['train', 'val']
for subset in subsets:
    if not os.path.isdir(os.path.join(subject_ds, subset)):
        os.mkdir(os.path.join(subject_ds, subset))

_, src_fps = recursive_file_retrieval(src_ds)

for fp in src_fps:
    
    if fp.endswith('.npy'):
        fn = fp.split('/')[-1]
        subset = fp.split('/')[-3]
        voice_dir = fp.split('/')[-2]
        
        trg_voice_dir = os.path.join(subject_ds, subset, voice_dir)
        trg_fp = os.path.join(trg_voice_dir, fn)
        if os.path.exists(trg_fp):
            continue
        os.mkdir(trg_voice_dir)
        os.rename(os.path.join(subject_ds, voice_dir), trg_voice_dir)


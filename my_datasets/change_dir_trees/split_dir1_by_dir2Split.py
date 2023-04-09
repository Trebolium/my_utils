import sys, os, pdb
from tqdm import tqdm
if os.path.abspath('..../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('..../my_utils'))
from my_os import recursive_file_retrieval

# dataset directories
ref_ds = sys.argv[1]
subject_ds = sys.argv[2]
ref_ext = sys.argv[3]
hidden_files_substr = '.0.'

# setup dir and tree
subsets = ['train', 'val', 'test']
for subset in subsets:
    if not os.path.isdir(os.path.join(subject_ds, subset)):
        os.mkdir(os.path.join(subject_ds, subset))

_, src_fps = recursive_file_retrieval(ref_ds)

for fp in tqdm(src_fps):
    # pdb.set_trace()
    
    if fp.endswith(ref_ext) and not hidden_files_substr in fp:
        try:
            fn = fp.split('/')[-1]
            subset = fp.split('/')[-3]
            voice_dir = fp.split('/')[-2]
            
            trg_voice_dir = os.path.join(subject_ds, subset, voice_dir)
            trg_fp = os.path.join(trg_voice_dir, fn)
            # pdb.set_trace()
            if os.path.exists(trg_fp.split('.')[0] +'.npy'):
                continue
            
            if not os.path.exists(trg_voice_dir):
                os.mkdir(trg_voice_dir)
            try:
                os.rename(os.path.join(subject_ds, voice_dir), trg_voice_dir)
            except:
                pdb.set_trace()
        except:
            pdb.set_trace()

        # if os.path.exists(trg_fp):
        #     continue
        # os.mkdir(trg_voice_dir)
        # os.rename(os.path.join(subject_ds, voice_dir), trg_voice_dir)
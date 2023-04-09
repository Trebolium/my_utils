import os, pickle, sys, shutil, pdb

# for some reason there is an unwatend path in sys.path. Must figure out how to remove this
for i in sys.path:
    if i == '/homes/bdoc3/wavenet_vocoder':
        sys.path.remove(i)

if os.path.abspath('.../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('.../my_utils'))
from my_os import recursive_file_retrieval

voice_id_list = pickle.load(open('missing_voice.pkl','rb'))
ref_ds = 'path/to/ds'
dst_dir = 'path/to/dst'

_, fps = recursive_file_retrieval(ref_ds)
fps = [fp for fp in fps if fp.endswith('wav')]

for vid in voice_id_list:
    vid_dir = os.path.join(ref_ds, vid)
    shutil.copyfile(vid_dir, dst_dir)
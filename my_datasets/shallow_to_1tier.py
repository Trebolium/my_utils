import os, pdb, shutil, sys
if os.path.abspath('..../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('..../my_utils'))
from my_os import recursive_file_retrieval

# pdb.set_trace()
src_dir = sys.argv[1]
f_ext = sys.argv[2]

_, src_fpaths = recursive_file_retrieval(src_dir)
# src_fpaths = [f for f in src_fpaths if f.endswith('.npy')]

singer_list = []
pdb.set_trace()
for f in src_fpaths:
    f_name = os.path.basename(f)
    if f.endswith(f_ext) and not f.startswith('.'):
        singer_id = f_name.split('_')[0]
        if singer_id not in singer_list:
            singer_list.append(singer_id)
            os.mkdir(os.path.join(src_dir, singer_id))
        shutil.move(os.path.join(src_dir, f_name), os.path.join(src_dir, singer_id, f_name))
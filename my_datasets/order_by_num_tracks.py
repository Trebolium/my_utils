import os, pdb

import sys
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval


src_dir = '/import/c4dm-02/bdoc3/spmel/damp_qianParams/train'

subdirs, all_fps = recursive_file_retrieval(src_dir)
my_list = []
for sd_path in subdirs[1:]:
    sdn = os.path.basename(sd_path)
    num = len(os.listdir(sd_path))
    my_list.append((num, sdn))

my_list.sort(key=lambda x: x[0])


pdb.set_trace()
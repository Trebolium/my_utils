
import sys, os, pickle
from my_os import recursive_file_retrieval

def ds_subpaths_to_pickle(src_ds, ext, name):

    src_ds = sys.argv[1]
    _, src_fps = recursive_file_retrieval(src_ds)
    src_fps = [fp for fp in src_fps if not fp.startswith('.') and fp.endswith(ext) and not '_size' in fp]
    with open(os.path.join(src_ds, name +'.pkl'), 'wb') as f:
        pickle.dump(src_fps, f)
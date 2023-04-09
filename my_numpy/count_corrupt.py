import numpy as np
import sys, pickle
from tqdm import tqdm
if os.path.abspath('.../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('.../my_utils'))
from my_os import recursive_file_retrieval

"""Counts corrupt numpy files in dir and saves list as pickle"""

use_multiprocess = False

diry = sys.argv[1]

_, all_fps = recursive_file_retrieval(diry)
all_fps = [path for path in all_fps if path.endswith('npy')]

errors = []
num_ers = 0

for fp in tqdm(all_fps):
    try:
        np.load(fp)
    except Exception as e:
        num_ers += 1
        print(f'Error num: {num_ers}, Exception: {e}, file_path: {fp}')
        errors.append((e, fp))

for e in errors:
    print(e)

filehandler = open('corrupt_numpy.pkl', 'wb')
pickle.dump(errors, filehandler)
print(f'number of numpy issues: {len(errors)}')
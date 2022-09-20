import numpy as np
from tqdm import tqdm
import sys, pdb, pickle
import concurrent.futures
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval

"""Converts arrays from 64 to 32
    If they cannot be opened, save them in a pickle file called 'corrupt_numpy.pkl'
    This list can be used later to reload the offending numpy arrays"""

def convert_64_to_32(all_fps):
    for path in (all_fps):
        try:
            arr64 = np.load(path, allow_pickle=True)
            if type(arr64) == np.int32:
                continue    
            arr32 = arr64.astype(np.int32)
            np.save(path, arr32)
            # print(f'{path} complete')
        except Exception as e:
            print(f'Exception: {e} caused by path {path}')
            return e, path


if __name__ == '__main__':

    ds_path = sys.argv[1]
    num_processes = int(sys.argv[2])

    exceptions = []
    _, all_fps = recursive_file_retrieval(ds_path)
    all_fps = [path for path in all_fps if path.endswith('npy')]

    for i in tqdm(range(0, len(all_fps), num_processes)):
        all_fps_chunk = all_fps[i:i+num_processes]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_objects = executor.map(convert_64_to_32, [all_fps_chunk])
            for fo in future_objects:
                if not fo == None:
                    exceptions.append(fo.result())


    for e in exceptions:
        print(e)

    filehandler = open('corrupt_numpy.pkl', 'wb')
    pickle.dump(exceptions, filehandler)
    print(f'number of numpy issues: {len(exceptions)}')
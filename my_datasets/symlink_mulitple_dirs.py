import pdb
import subprocess
import os
import math
import sys
sys.path.insert(1, '/homes/bdoc3/my_utils') # only need to do this once in the main script
from my_os import recursive_file_retrieval
from my_csv import vctk_id_gender_list
from tqdm import tqdm


"""
This assumes the usual datset format of 3 subsets of dirs of data files.
It takes the first (currently not randomised) defined percentage of these files and creates symlinks for them to be used in a hidden directory.
This allows flexibility to define how much of a dataset you want to use, as long as you predefine it here first
"""

ds_dir = sys.argv[1]
ds_fraction = float(sys.argv[2])
ds_fraction_dir =  '.' +sys.argv[2] +'_size'

for subset in ['train', 'val', 'test']:
    print(f'Processing files in {subset}')
    hidden_fractional_dir = os.path.join(ds_dir, subset, ds_fraction_dir)
    if not os.path.exists(hidden_fractional_dir):
        os.mkdir(hidden_fractional_dir)

    sdps, fps = recursive_file_retrieval(os.path.join(ds_dir, subset), ignore_hidden_dirs=True)
    sdps = sdps[1:] # ignore first entry which is original parent path
    # take the first fraction - not random - only in order to suuposedly reproduce Qians results
    sdps = sdps[:round(len(sdps)*ds_fraction)]

    for sdp in tqdm(sdps):

        bashCommand = f"ln -s {sdp}/ {hidden_fractional_dir}/"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


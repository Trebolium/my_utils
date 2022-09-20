import sys, os, pdb
from tqdm import tqdm
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval

# reorder one directory so that its files are in the same directory tree as a reference directory
def reorder_dir_by_another(ref_dir, reorder_dir):

    _, reorder_fps = recursive_file_retrieval(reorder_dir)
    reorder_fps = [fp for fp in reorder_fps if not fp.startswith('.') and fp.endswith('npy')]

    subsets = ['train', 'val', 'test']
    missing_from_ref = []
    pdb.set_trace()
    
    for fp in tqdm(reorder_fps):
        print(f'Searching for twin of {fp}')
        if 'dev' in fp:
            raise Exception('Detected subset called dev. Convert this to train and try again')

        fn = os.path.basename(fp)
        original_voice_dir = os.path.dirname(fp)
        file_found = False

        for subset in subsets:
            possible_ref_path = os.path.join(ref_dir, subset, fn.split('_')[0], fn)

            # when path found, send file to corresponding path branch in reorder_dir
            if os.path.exists(possible_ref_path):
                reordered_voice_dir = os.path.join(reorder_dir, subset, fn.split('_')[0])
                reordered_fp = os.path.join(reorder_dir, subset, fn.split('_')[0], fn)
                if not os.path.exists(reordered_voice_dir):
                    os.makedirs(reordered_voice_dir)    
                os.rename(fp, reordered_fp)
                # delete if remaining subdir is empty
                if len(os.listdir(original_voice_dir)) == 0:
                    os.rmdir(original_voice_dir)

                file_found = True
                break

        if file_found == False:
            print('MISSING: Couldnt find a match')
            missing_from_ref.append(fp)
            continue

    for i, fp in enumerate(missing_from_ref):
        print(i, fp)

    pdb.set_trace()


# check to see whether two directories have the same labelled files in the same relative directory trees
def identical_dir_contents(ref_dir, reorder_dir):

    _, ref_fps = recursive_file_retrieval(ref_dir)
    rel_ref_fps = [os.path.relpath(fp, ref_dir) for fp in ref_fps if not fp.startswith('.') and fp.endswith('npy')]

    _, reorder_fps = recursive_file_retrieval(reorder_dir)
    rel_reorder_fps = [os.path.relpath(fp, reorder_dir) for fp in reorder_fps if not fp.startswith('.') and fp.endswith('npy')]

    same = True
    for fp in rel_reorder_fps:
        if fp not in rel_ref_fps:
            same = False
            break

    return same

if __name__ == '__main__':

    ref_dir = sys.argv[1]
    reorder_dir = sys.argv[2]
    reorder_dir_by_another(ref_dir, reorder_dir)
    print(f'Is the reordered directory now the same as the reference directory?: {identical_dir_contents(ref_dir, reorder_dir)}')

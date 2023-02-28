from asyncore import file_dispatcher
import os, sys, random, shutil, math, pdb
from tqdm import tqdm
from my_interaction import binary_answer

# as it says on the tin
def overwrite_dir(directory, ask):
    if os.path.exists(directory):
        if ask:
            print(f'Directory name {directory} already exists. Would you like to overwrite it?')
            answer = binary_answer()
            if answer:
                shutil.rmtree(directory)
            else:
                print('Try a new file name and re-run this program')
                exit(0)
        else:
            shutil.rmtree(directory)
    os.makedirs(directory)


# returns a list of filepaths collected from a parent directory and all subdirectories
def recursive_file_retrieval(parent_path, ignore_hidden_dirs=False, return_parent=True):
    
    file_path_list = []
    dir_list = []
    parent_paths = [parent_path]

    more_subdirs = True
    while more_subdirs == True:
        subdir_paths = []
        for i, parent_path in enumerate(parent_paths):
            # print(parent_path)
            if ignore_hidden_dirs:
                if os.path.basename(parent_path).startswith('.'):
                    continue

            dir_list.append(parent_path)
            r,dirs,files = next(os.walk(parent_path, topdown=True, onerror=None, followlinks=False)) 
            for f in files:
                file_path_list.append(os.path.join(r,f))

            # if there are more subdirectories
            if len(dirs) != 0:
                for d in dirs:
                    subdir_paths.append(os.path.join(r,d))

            # if we've finished going through subdirectories (each parent_path), stop that loop
            if i == len(parent_paths)-1:
                # if loop about to finish, change parent_paths content and restart loop
                if len(subdir_paths) != 0:
                    parent_paths = subdir_paths
                else:
                    more_subdirs = False
    
    if not return_parent: dir_list = dir_list[1:]

    return dir_list, file_path_list


# randomly choses file paths from a list and copy them to a destination folder
def copy_random_files(tracklist, dst_dir, num_choices):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for choice in range(num_choices):
        chosen_path = random.choice(tracklist)
        shutil.copy(chosen_path, dst_dir + '/' +os.path.basename(chosen_path))

# if the save_path already exists, make a new copy
def save_rename_duplicate(save_path):
    save_num = 1
    while os.path.exists(save_path):
            save_num += 1
            save_path = save_path +'_copy' +str(save_num)
    return save_path

# generate a destination path for the current file (within a dataset train/val directory)
def get_dst_file_path(file_path, dst_dir, is_val_set):
    singer_id = os.path.basename(file_path).split('_')[0]
    if not is_val_set:
        voice_dir_path = os.path.join(dst_dir, 'train', singer_id)
    else:
        voice_dir_path = os.path.join(dst_dir, 'val', singer_id) 
    if not os.path.exists(voice_dir_path):
        os.mkdir(voice_dir_path)
    dst_file_path = os.path.join(voice_dir_path, os.path.splitext(os.path.basename(file_path))[0]+'.npy')
    return dst_file_path

# split by ratio, but check for repeating speakers straddling the splitting point of train and val sets
def generate_voice_subsets(path_list, train_split):
    path_list.sort()
    split_int = math.floor(len(path_list)*train_split)
    last_train_singer = os.path.basename(path_list[split_int - 1]).split('_')[0]
    first_val_singer = os.path.basename(path_list[split_int]).split('_')[0]

    while last_train_singer == first_val_singer:
        split_int -= 1
        last_train_singer = os.path.basename(path_list[split_int - 1]).split('_')[0]
        first_val_singer = os.path.basename(path_list[split_int]).split('_')[0]
    
    train_paths, val_paths = path_list[:split_int], path_list[split_int:]
    return [train_paths, val_paths]

# splits an unsplit dataset into train and val subsets
def split_dataset_subsets(dataset_path, split_ratio, relevant_ext, seed_int=1, using_subdirs=True):
    # collect relevant items
    random.seed(seed_int)

    all_dirs_paths, all_file_paths = recursive_file_retrieval(dataset_path)
    # make subset dirs
    if not os.path.exists(os.path.join(dataset_path, 'train')):
        os.mkdir(os.path.join(dataset_path, 'train'))
    if not os.path.exists(os.path.join(dataset_path, 'val')): 
        os.mkdir(os.path.join(dataset_path, 'val'))
    if using_subdirs:
        all_dirs_paths = [path for path in all_dirs_paths if not os.path.basename(path).startswith('.')][1:] # exclude hidden folders
        items = all_dirs_paths
    else:
        all_file_paths = [path for path in all_file_paths if os.path.basename(path).endswith(relevant_ext)] # inly include files with relevant extension
        items = all_file_paths
    
    random.shuffle(items)

    num_train_items = math.floor(split_ratio*len(items))
    for i in range(len(items)):
        if i <= num_train_items:
            print(i, items[i], os.path.join(dataset_path, 'train', os.path.basename(items[i])))
            os.rename(items[i], os.path.join(dataset_path, 'train', os.path.basename(items[i])))
        else:
            os.rename(items[i], os.path.join(dataset_path, 'val', os.path.basename(items[i])))

# combine subsets to form one subsetless dataset
def consolidate_dataset_subsets(dataset_path):
    root_dir = os.path.basename(dataset_path)
    all_dirs_paths, all_file_paths = recursive_file_retrieval(dataset_path)
    items = [dir_path for dir_path in all_dirs_paths if not dir_path.endswith((root_dir,'train','val')) and ('val' in dir_path or 'train' in dir_path)]
    pdb.set_trace()
    for i in tqdm(range(len(items))):
        try:
            if os.path.exists(os.path.join(dataset_path, os.path.basename(items[i]))):
                _, file_paths = recursive_file_retrieval(items[i])
                for f_path in file_paths:
                    os.rename(f_path, os.path.join(dataset_path, os.path.basename(items[i]), os.path.basename(f_path)))
                os.rmdir(items[i])
            else:
                os.rename(items[i], os.path.join(dataset_path, os.path.basename(items[i])))
        except Exception as e:
            print(e)
            pdb.set_trace()
    os.rmdir(os.path.join(dataset_path, 'train'))
    os.rmdir(os.path.join(dataset_path, 'val'))

# Any files from dir1 that aren't in dir2 are deleted
def delete_uncommon_files(dir1, dir2):
    dir1dirs, dir1files = recursive_file_retrieval(dir1)
    dir2dirs, dir2files = recursive_file_retrieval(dir2)
    dir2basenames = [os.path.basename(i) for i in dir2files]
    dir1files_not_in_dir2 = [file for file in dir1files if os.path.basename(file) not in dir2basenames]
    pdb.set_trace()
    for f_path in dir1files_not_in_dir2:
        os.remove(f_path)
        if len(os.listdir(os.path.dirname(f_path))) == 0:
            os.rmdir(os.path.dirname(f_path))

# def ds_to_subsets:

if __name__ == '__main__':
    split_dataset_subsets('/homes/bdoc3/my_data/audio_data/deslienced_concat_DAMP', 0.8, '.wav')
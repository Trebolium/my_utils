import os, math, pdb, yaml

# generate a destination path for the current file (within a dataset train/val directory)
def get_dst_file_path(file_path, dst_dir, is_val_set):
    singer_id = os.path.basename(file_path).split('_')[0]
    if not is_val_set:
        voice_dir_path = os.path.join(dst_dir, 'train', singer_id)
    else:
        voice_dir_path = os.path.join(dst_dir, 'val', singer_id) 
    if not os.path.exists(voice_dir_path):
        os.makedirs(voice_dir_path)
    dst_file_path = os.path.join(voice_dir_path, os.path.splitext(os.path.basename(file_path))[0]+'.npy')
    return dst_file_path


#### datasets ####

# split by ratio, but check for repeating speakers straddling the splitting point of train and val sets
def split_subsets_by_subdirs(path_list, train_split):
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

# create and populate the dataset directory
def make_dataset_dir(dst_dir, feat_params):
    if not os.path.exists(dst_dir):
        os.makedirs(os.path.join(dst_dir, 'train'))
        os.mkdir(os.path.join(dst_dir, 'val'))
    with open(os.path.join(dst_dir, 'feat_params.yaml'), 'w') as handle:
        yaml.dump(feat_params, handle)

# returns a list of filepaths collected from a parent directory and all subdirectories
def recursive_file_retrieval(parent_path):
    file_path_list = []
    dir_list = []
    parent_paths = [parent_path]
    more_subdirs = True
    while more_subdirs == True:
        subdir_paths = [] 
        for i, parent_path in enumerate(parent_paths):
            dir_list.append(parent_path)
            r,dirs,files = next(os.walk(parent_path)) 
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
    return dir_list, file_path_list
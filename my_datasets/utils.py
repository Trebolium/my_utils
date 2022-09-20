import os, math, pdb, yaml, random

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
def make_dataset_dir(dst_dir, feat_params, auto_subsets=False):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        if auto_subsets:
            os.mkdir(os.path.join(dst_dir, 'train'))
            os.mkdir(os.path.join(dst_dir, 'val'))
    with open(os.path.join(dst_dir, 'feat_params.yaml'), 'w') as handle:
        yaml.dump(feat_params, handle)


# if ds is already divided by spkr dirs, use this to split into subsets
def create_ds_subsets(ds_dir, split=0.8):
    pdb.set_trace()
    _, sub_dirs, _ = next(os.walk(ds_dir))
    train_list = random.sample(sub_dirs, k=math.floor(len(sub_dirs)*split))
    test_list = [sub_dir for sub_dir in sub_dirs if sub_dir not in train_list]

    if not os.path.exists(os.path.join(ds_dir, 'train')): os.mkdir(os.path.join(ds_dir, 'train'))
    if not os.path.exists(os.path.join(ds_dir, 'val')): os.mkdir(os.path.join(ds_dir, 'val'))

    for sub_dir in train_list:
        os.rename(os.path.join(ds_dir, sub_dir), os.path.join(ds_dir, 'train', sub_dir))
    for sub_dir in test_list:
        os.rename(os.path.join(ds_dir, sub_dir), os.path.join(ds_dir, 'val', sub_dir))

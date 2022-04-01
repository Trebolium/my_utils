from copy import Error
import numpy as np
import sys, os, torch, shutil, pdb
from tqdm import tqdm
sys.path.insert(1, '\homes\bdoc3\my_utils')
from my_os import recursive_file_retrieval
from my_normalise import TorchStatsRecorder

def normalise_data(src_path, dst_path, del_src_ds=False):
    " Provide src and dst dir paths in full"
    root_dst_path = '/'.join(dst_path.split('/')[:-2]) if dst_path.endswith('/') else '/'.join(dst_path.split('/')[:-1])
    dst_dir = dst_path.split('/')[-2] if dst_path.endswith('/') else dst_path.split('/')[-1]

    _, file_path_list = recursive_file_retrieval(src_path)
    for file_p in file_path_list:
        dir_path = os.path.dirname(file_p)
        if dir_path.endswith('val') or dir_path.endswith('test'):
            raise Exception("Directory named val or test found in dataset. Only training set should be used for normalisation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_rec = TorchStatsRecorder(red_dims=0)
    error_num = 0
    error_list = []
    
    for f_path in tqdm(file_path_list):
        if f_path.endswith('.yaml'): continue
        try:
            feats = np.load(f_path)
        except ValueError as e:
            error_num +=1
            print(f'File path {f_path} gives error: {e}')
            error_list.append((f_path, e))
            continue
        torch_feats = torch.from_numpy(feats).float().to(device)
        stats_rec.update(torch_feats)

    total_mean, total_std = stats_rec.mean, stats_rec.std

    print(f'Total dataset stats: \n\n Mean: {total_mean} \n Std: {total_std}')
    print('\n Now generating new normalised dataset...')
    for f_path in tqdm(file_path_list):
        if f_path.endswith('.yaml'): continue
        file_name = os.path.basename(f_path)
        try:
            feats = np.load(f_path)
        except ValueError as e:
            continue
        torch_feats = torch.from_numpy(feats).to(device)
        normed_feats = ((torch_feats - total_mean) / total_std).cpu().numpy()
        file_dst_path = os.path.join(root_dst_path, dst_dir, os.path.relpath(f_path, src_path))
        if not os.path.exists(os.path.dirname(file_dst_path)):
            os.makedirs(os.path.dirname(file_dst_path))
        np.save(file_dst_path, normed_feats)

    if del_src_ds:
        shutil.rmtree(src_path)

    for i in error_list: print(i)
    print(f'Number of errors encountered: {error_num}')


if __name__ == '__main__':
    normalise_data(sys.argv[1], sys.argv[2], del_src_ds=False) 
import numpy as np
import torch, pdb
from my_os import recursive_file_retrieval
from tqdm import tqdm

"""
This script normalises data across all features. In the case of audio spectrograms,
each frequency band can be thought of as an individual feature, and so the data is normalised across each frequency band separately.
The code for this was provided in http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html,
and adapted in https://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html
"""

class StatsRecorder:
    def __init__(self, stats_dim, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        self.stats_dim = stats_dim
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=self.stats_dim)
            self.std  = data.std(axis=self.stats_dim)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: torch, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(self.stats_dim, data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=self.stats_dim)
            newstd  = data.std(axis=self.stats_dim)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

class TorchStatsRecorder:
    def __init__(self, red_dims=1):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n


"""
    Goes through a dataset directory (just the partition for training,
    And collects mean and std info across a specified dimension
    of the feature arrays stored within it
    Returns the total mean and std
"""
def get_norm_stats(src_path, num_feats=None, which_cuda=0, red_dims=0, report_errs=False):

    _, file_path_list = recursive_file_retrieval(src_path)

    # ensure val and test dirs are not within path
    for file_p in file_path_list:
        if 'val' in file_p or 'test' in file_p or 'dev' in file_p:
            raise Exception("Directory named val or test found in dataset. Only training set should be used for normalisation")

    device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else "cpu")
    stats_rec = StatsRecorder(stats_dim=red_dims)
    error_list = []
    
    for f_path in tqdm(file_path_list):
        if not f_path.endswith('.npy'): continue
        try:
            feats = np.load(f_path)
        except ValueError as e:
            error_list.append((f_path, e))
            continue
        # feats = torch.from_numpy(feats).float().to(device)
        if num_feats != None:
            feats = feats[:,:num_feats]
        stats_rec.update(feats)

    total_mean, total_std = stats_rec.mean, stats_rec.std
    # print(f'Total dataset stats: \n\n Mean: {total_mean} \n Std: {total_std}')
    if report_errs:
        return total_mean, total_std
    else:
        return total_mean, total_std, error_list


# apply stats data to incoming example
def apply_norm_stats(feats, total_mean, total_std, which_cuda=0):

    device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else "cpu")
    feats = torch.from_numpy(feats).to(device)
    normed_feats = ((feats - total_mean) / total_std).cpu().numpy()

    return normed_feats


# map array elements to limits [0,1]
def zero_one_mapped(arr):
    grounded_arr = arr - np.min(arr)
    normed_arr = grounded_arr / np.max(grounded_arr)
    return normed_arr

# normalize array to be unit variance
def unit_var(arr):
    return (arr - np.mean(arr)) / np.std(arr)

def schluter(arr, stats):
    # pdb.set_trace()
    total_mean, total_std = stats[0], stats[1]
    if arr.shape[1] != len(total_mean) or arr.shape[1] != len(total_std):
        raise Exception('Schluter Norm Error: Mismatch between array and stats dimensions')
    normed_arr = (arr - total_mean) / total_std
    return normed_arr

def guv(arr, stats):
    freq_total_means, freq_total_stds = stats[0], stats[1]
    global_mean = np.mean(freq_total_means)
    global_std = np.sqrt(np.sum(np.square(freq_total_stds))/len(freq_total_stds)) #https://www.statology.org/averaging-standard-deviations/, https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    return (arr - global_mean) / global_std

def norm_feat_arr(arr, norm_method, stats=None):
    if norm_method == 'zero_one':
        arr = zero_one_mapped(arr)
    elif norm_method == 'unit_var':
        arr = unit_var(arr)
    elif norm_method == 'global_unit_var':
        arr = guv(arr, stats)
    elif norm_method == 'schluter':
        if stats==None:
            raise Exception('Schluter Norm Error: Requires stats to be applied to data')
        arr = schluter(arr, stats)
    elif norm_method == None:
        pass
    else:
        raise Exception('Norm method not recognised.')
    return arr
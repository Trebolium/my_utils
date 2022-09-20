import matplotlib.pyplot as plt
import torch
import numpy as np

def save_spec(arr, filename, dim1_is_time=True, colorbar=True):
    if len(arr.shape) != 2:
        Exception('Tried to print spectrogram from array that is not 2 dims')
    # check if tensor, int type, list
    if type(arr) == torch.Tensor:
        arr = arr.cpu().detach().numpy()
    elif type(arr) == list:
        arr = np.asarray(arr)
    if dim1_is_time:
        arr = np.rot90(arr)
    plt.imshow(arr)
    plt.colorbar()
    plt.savefig(filename)
    

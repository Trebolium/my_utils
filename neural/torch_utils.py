import numpy as np
import torch

def container_to_tensor(container, add_batch_dim=False, device='cpu'):
    
    if type(container) == list:
        container = np.asarray(container)
    if type(container) == np.ndarray:
        container = torch.from_numpy(container).float()

    if add_batch_dim:
        container = container.unsqueeze(0)
    container = container.to(device)
    return container


def tensor_to_array(tensor):
    squeezed_tensor = torch.squeeze(tensor)
    arr = squeezed_tensor.detach().cpu().numpy()
    return arr
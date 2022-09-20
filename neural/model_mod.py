from collections import OrderedDict
import torch

# to set the entire model and its tensors to a device, just use 'model.to(device)'

def set_optimizer_device(optimizer, device):
    # fixes tensors on different devices error
    # https://github.com/pytorch/pytorch/issues/2830
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer


def checkpoint_model_optim_keys(checkpoint):
    for k in checkpoint.keys():
        if k.startswith('model'):
            model_key = k
        if k.startswith('optim'):
            optim_key = k
    return model_key, optim_key

# could possibly instead use 'model.load_state_dict(g_checkpoint['model_state_dict'])'
def ckpt_to_model(ckpt, key, model):
    new_state_dict = OrderedDict()
    for k, v in ckpt[key].items():
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    return model

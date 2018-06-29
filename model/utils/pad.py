import torch
from torch.autograd import Variable
from .convert import to_var


def pad(tensor, length):
    print("Pad, type: ", get_type())
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).type(get_type())])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).type(get_type())])
        else:
            return tensor


def get_type():
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor
    return torch.FloatTensor


def pad_and_pack(tensor_list):
    length_list = ([t.size(0) for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = torch.stack(padded, 0)
    return packed, length_list

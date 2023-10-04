import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from data_util import cifar_class_idx

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def dsa(ds, k, verbose = False):
    tot = 0
    acc = 0
    for batch in ds:
        out = k(batch[0].view(-1, 28*28))
        acc += accuracy(out, batch[1]) * len(batch[1])
        tot += len(batch[1])
    if verbose:
        print(acc, tot)
    return (acc/tot).item()

def asr(ds, k, adv_alg = None, verbose = False, **kwargs):
    tot = 0
    acc = 0
    for batch in (tqdm(ds) if verbose else ds):
        if adv_alg is None:
            out = k(batch[0])
        else:
            out = k(batch[0] + adv_alg(batch[0], batch[1], k, **kwargs))
        acc += accuracy(out, batch[1]) * len(batch[1])
        tot += len(batch[1])
    return (1 - acc/tot).item()

def scale_im(im):
    return (im - im.min())/(im.max() - im.min())

def expand_first(im):
    return im.reshape(1, *im.shape)

def gen_base_affine(x, device = torch.device('cuda:0')):
    thetas = torch.zeros(x.shape[0], 2, 3)
    thetas[:, 0, 0] = 1
    thetas[:, 1, 1] = 1
    return thetas.to(device)

def diff_affine(x, thetas):
    base_theta = gen_base_affine(x)
    grid = F.affine_grid(base_theta + thetas, x.size(), align_corners = False)
    return F.grid_sample(x, grid, align_corners = False)

def project_lp(v, norm, xi, exact = False, device = device):
    if v.dim() == 4:
        batch_size = v.shape[0]
    else:
        batch_size = 1
    if exact:
        if norm == 2:
            if batch_size == 1:
                v = v * xi/torch.norm(v, p = 2)
            else:
                v = v * xi/torch.norm(v, p = 2, dim = (1,2,3)).reshape((batch_size, 1, 1, 1))
        elif norm == np.inf:
            v = torch.sign(v) * torch.minimum(torch.abs(v), xi*torch.ones(v.shape, device = device))
        else:
            raise ValueError('L_{} norm not implemented'.format(norm))
    else:
        if norm == 2:
            if batch_size == 1:
                v = v * torch.minimum(torch.ones((1), device = device), xi/torch.norm(v, p = 2))
            else:
                v = v * torch.minimum(xi/torch.norm(v, p = 2, dim = (1,2,3)), torch.ones(batch_size, device = device)).reshape((batch_size, 1, 1, 1))
        elif norm == np.inf:
            v = torch.sign(v) * torch.minimum(torch.abs(v), xi*torch.ones(v.shape, device = device))
        else:
            raise ValueError('L_{} norm not implemented'.format(norm))
    return v
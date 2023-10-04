import torch
from util import project_lp, diff_affine
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def fgsm(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = torch.device('cuda:0')):
    v = torch.zeros_like(x, device = device, requires_grad = True)
    loss = F.cross_entropy(k(x + v), y)
    loss.backward()
    return project_lp(step_size * torch.sign(v.grad), norm = norm, xi = xi, device = device)

def pgd(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-2, epochs = 40, random_restart = 4, device = torch.device('cuda:0')):
    batch_size = x.shape[0]
    max_loss = F.cross_entropy(k(x), y)
    max_X = torch.zeros_like(x)
    random_delta = torch.rand(size = (batch_size * random_restart, *x.shape[1:]), device = device) - 0.5
    random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
    x = x.repeat(random_restart, 1, 1, 1)
    y = y.repeat(random_restart)
    for j in range(epochs):
        v = torch.zeros_like(random_delta, device = device, requires_grad = True)
        loss = F.cross_entropy(k(x + random_delta + v), y)
        loss.backward()
        pert = step_size * torch.sign(v.grad)#torch.mean(v.grad)
        random_delta = project_lp(random_delta + pert, norm = norm, xi = xi, device = device)
    _,idx = torch.max(F.cross_entropy(k(x + random_delta), y, reduction = 'none').reshape(random_restart, batch_size), axis = 0)
    return random_delta[idx * batch_size + torch.arange(batch_size, dtype = torch.int64, device = device)]

def batch_pgd(x, y, k, norm = 2, xi = 5, epochs = 40, random_restart = 4, step_size = 1e-2, decay = 0.95, device = device):
    batch_size = x.shape[0]
    random_delta = torch.rand(size = (random_restart, *x.shape[1:]), device = device) - 0.5
    random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
    random_delta.requires_grad = True
    x = x.repeat(random_restart, 1, 1, 1)
    y = y.repeat(random_restart)
    for j in range(epochs):
        pert_x = x + random_delta.repeat_interleave(batch_size, dim = 0)
        loss = F.cross_entropy(k(pert_x), y)
        loss.backward()
        pert = step_size * torch.sign(random_delta.grad)
        step_size = step_size * decay
        random_delta = project_lp(random_delta.detach() + pert, norm = norm, xi = xi, device = device)
        random_delta.requires_grad = True
    _,idx = torch.max(F.cross_entropy(mdl(x + random_delta.repeat_interleave(batch_size, dim = 0)), y, reduction = 'none').reshape(random_restart, batch_size).sum(axis = 1), axis = 0)
    return random_delta[idx:(idx+1)]
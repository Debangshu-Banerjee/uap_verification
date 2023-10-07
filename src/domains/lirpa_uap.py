import torch
import numpy as np
import random

from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from auto_LiRPA.utils import get_spec_matrix
from cert_util import min_correct_with_eps, load_data, DeltaWrapper

from model_defs import mnist_cnn_4layer,mnist_conv_small,mnist_conv_big


def bounded_results(eps,bounded_model):
    ptb = PerturbationLpNorm(norm = np.inf, eps = eps)
    bounded_delta = BoundedTensor(delta, ptb)

    final_name = bounded_model.final_name
    input_name = '/1' # '/input.1' 

    result = bounded_model.compute_bounds(
        x=(new_image,bounded_delta), method='backward', C=C,
        return_A=True, 
        needed_A_dict={ final_name: [input_name] },
    )
    lower, upper, A_dict = result
    lA = A_dict[final_name][input_name]['lA']
    uA = A_dict[final_name][input_name]['uA']

    lb = lower - ptb.concretize(delta, lA, sign=-1)
    ub = upper - ptb.concretize(delta, uA, sign=1)


    lA = torch.reshape(lA,(eval_num, num_cls-1,-1))
    return lA,lb,lower
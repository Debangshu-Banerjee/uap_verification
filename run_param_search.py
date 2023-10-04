import subprocess

from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as analyzer
import src.uap_analyzer as uap_ver
from threading import Thread
import sys

# nets = [config.MNIST_CONV_MED, config.MNIST_CONV_SMALL, config.MNIST_LINEAR_50, config.MNIST_LINEAR_100, config.MNIST_FFN_PGD, config.MNIST_FFN_DIFFAI]
# epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:
#         subprocess.Popen(["python", "run_test.py", net, str(eps)])

# def run_test(eps, net, dataset = Dataset.MNIST):
#     uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=dataset, sink_label=None,
#         spec_type=InputSpecType.UAP, count=10, count_per_prop=5, eps=eps, net=net,                                                                                                              
#         timeout=100, output_dir='results/newer_uap/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True)
#     uap_ver.UapVerification(uap_verfication_args)

# nets = [config.MNIST_CONV_BIG]
# epsilons = [0.1, 0.2, 0.3, 0.4]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:
#         run_test(eps, net)

# nets = [config.CIFAR_CONV_SMALL, config.CIFAR_CONV_SMALL_PGD, config.CIFAR_CONV_SMALL_DIFFAI, config.CIFAR_CONV_MED]
# epsilons = [1/255, 2/255, 3/255, 4/255]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:
#         subprocess.Popen(["python", "run_test.py", net, str(eps)])

# nets = [config.MNIST_LINEAR_6_100, config.MNIST_LINEAR_9_200]#[config.MNIST_CONV_BIG]
# epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:
#         subprocess.Popen(["python", "run_test.py", net, str(eps)])

# def run_test(eps, net, dataset = Dataset.MNIST):
#     uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=dataset, sink_label=None,
#         spec_type=InputSpecType.UAP_BINARY, count=20, count_per_prop=15, eps=eps, net=net,                                                                                                              
#         timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True)
#     uap_ver.UapVerification(uap_verfication_args)

# nets = [config.MNIST_BINARY]
# epsilons = [0.05, 0.1, 0.15, 0.2, 0.25]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:
#         run_test(eps, net)

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.HOUSING, sink_label=None,
#         spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=50.0, net=config.HOUSING_RM_CRIM,                                                                                                              
#         timeout=100, output_dir='results/mono/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = config.MONOTONE_PROP.CRIM, monotone_inv=True)

# def run_test(eps, net, dataset = Dataset.MNIST, monotone_prop = False, monotone_inv = False):
#     uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=dataset, sink_label=None,
#         spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=eps, net=net,                                                                                                              
#         timeout=100, output_dir='results/mono_2layer_100/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = monotone_prop, monotone_inv=monotone_inv)
#     uap_ver.UapVerification(uap_verfication_args)

# props = [config.MONOTONE_PROP.RM, config.MONOTONE_PROP.CRIM]
# invs = [False, True]
# epsilons = [100.0, 150.0, 200.0, 250.0, 300.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
# for eps in epsilons:
#     run_test(eps, config.HOUSING_2LAYER_100, dataset= Dataset.HOUSING, monotone_prop = config.MONOTONE_PROP.RM, monotone_inv = False)

    
# epsilons = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
# #epsilons = [5.0]
# for eps in epsilons:
#     run_test(eps, config.HOUSING_2LAYER_100, dataset= Dataset.HOUSING, monotone_prop = config.MONOTONE_PROP.CRIM, monotone_inv = True)
    
def run_test(eps, net, dataset = Dataset.MNIST, monotone_prop = False, monotone_inv = False):
    uap_verfication_args = uap_ver.UapAnalysisArgs(
        individual_prop_domain=Domain.DEEPZ,
        domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=dataset, sink_label=None,
        spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=eps, net=net,                                                                                                              
        timeout=100, output_dir='results/mono_2layer_200/', radius_l=0.002, radius_r=0.25,
        uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
        no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = monotone_prop, monotone_inv=monotone_inv)
    uap_ver.UapVerification(uap_verfication_args)

props = [config.MONOTONE_PROP.RM, config.MONOTONE_PROP.CRIM]
invs = [False, True]
epsilons = [10.0, 20.0, 30.0]
for eps in epsilons:
    run_test(eps, config.HOUSING_2LAYER_200, dataset= Dataset.HOUSING, monotone_prop = config.MONOTONE_PROP.RM, monotone_inv = False)

    
# epsilons = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
# #epsilons = [5.0]
# for eps in epsilons:
#     run_test(eps, config.HOUSING_2LAYER_200, dataset= Dataset.HOUSING, monotone_prop = config.MONOTONE_PROP.CRIM, monotone_inv = True)


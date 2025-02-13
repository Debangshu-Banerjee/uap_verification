
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as analyzer
import src.uap_analyzer as uap_ver
from threading import Thread
import sys
import requests
from os import listdir
from os.path import isfile, join
import faulthandler

# faulthandler.enable()
#for eps in [0.000001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
for eps in [0.50]:
    for prop in [0, 2, 3, 4, 5]:
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPPOLY,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.ADULT, sink_label=None,
            spec_type=InputSpecType.UAP, count=20, count_per_prop=1, eps=eps, net=config.ADULT_TANH,                                                                                                              
            timeout=100, output_dir='mono_results_lp/', radius_l=0.002, radius_r=0.25, 
            uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = prop, monotone_inv=False)
        uap_ver.UapVerification(uap_verfication_args)

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#     individual_prop_domain=Domain.DEEPPOLY,
#     domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.ADULT, sink_label=None,
#     spec_type=InputSpecType.UAP, count=10, count_per_prop=1, eps=0.55, net=config.ADULT_TANH,                                                                                                              
#     timeout=100, output_dir='mono_results/', radius_l=0.002, radius_r=0.25, 
#     uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
#     no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = 0, monotone_inv=False)
# uap_ver.UapVerification(uap_verfication_args)


# uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.HOUSING, sink_label=None,
#         spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=50.0, net=config.HOUSING_RM_CRIM,                                                                                                              
#         timeout=100, output_dir='results/mono/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = config.MONOTONE_PROP.CRIM, monotone_inv=True)
# nets = [config.MNIST_CONV_MED, config.MNIST_CONV_SMALL, config.MNIST_LINEAR_50, config.MNIST_LINEAR_100, config.MNIST_FFN_PGD, config.MNIST_FFN_DIFFAI]
# epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
# print("-------Starting all Threads-------")
# for net in nets:
#     for eps in epsilons:

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#     individual_prop_domain=Domain.DEEPZ,
#     domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
#     spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=float(sys.argv[2]), net=sys.argv[1],                                                                                                              
#     timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
#     uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#     no_lp_for_verified = True, debug_mode=False, track_differences=True)
# thread = Thread(target=uap_ver.UapVerification, args=(uap_verfication_args,))
# thread.start()

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#     individual_prop_domain=Domain.DEEPZ,
#     domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
#     spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=float(sys.argv[2]), net=sys.argv[1],                                                                                                              
#     timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
#     uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#     no_lp_for_verified = True, debug_mode=False, track_differences=True)
# thread = Thread(target=uap_ver.UapVerification, args=(uap_verfication_args,))
# thread.start()

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#     individual_prop_domain=Domain.DEEPZ,
#     domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
#     spec_type=InputSpecType.UAP, count=3, count_per_prop=5, eps=float(sys.argv[2]), net=sys.argv[1],                                                                                                              
#     timeout=100, output_dir='results/param_search/', radius_l=0.002, radius_r=0.25,
#     uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#     no_lp_for_verified = True, debug_mode=False, track_differences=True)
# thread = Thread(target=uap_ver.UapVerification, args=(uap_verfication_args,))
# thread.start()

# eps = 0.1
# for _ in range(4):
#     uap_verfication_args = uap_ver.UapAnalysisArgs(>
#         individual_prop_domain=Domain.DEEPPOLY,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
#         spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CONV_SMALL,                                                                                                              
#         timeout=100, output_dir='par_results/', radius_l=0.002, radius_r=0.25,
#         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
#         try_image_smoothing=False, filter_threshold=None)
#     uap_ver.UapVerification(uap_verfication_args)
#     eps += 0.05

# net = config.CIFAR_CROWN_IBP_MEDIUM
# search_eps = [1.0, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75]
# epsilons = []
# files = [f for f in listdir('./cifar_results/') if isfile(join('./cifar_results/', f))]
# for se in search_eps:
#     f_name = '{}_5_20_{:.2f}_Domain.DEEPZ.dat'.format(net, se)
#     if not f_name in files:
#         epsilons.append(se)

# if len(epsilons) > 0:
#     print(f'Evaluating {net} at {epsilons}')
#     for eps in epsilons:
#         uap_verfication_args = uap_ver.UapAnalysisArgs(
#             individual_prop_domain=Domain.DEEPZ,
#             domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
#             spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=net,                                                                                                              
#             timeout=100, output_dir='cifar_results/', radius_l=0.002, radius_r=0.25, 
#             uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#             no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
#             try_image_smoothing=False, filter_threshold=None, ligweight_diffpoly = True)
#         uap_ver.UapVerification(uap_verfication_args)
#         requests.post('https://ntfy.cmxu.io/test', data = f'Finished {net} at {eps}'.encode(encoding='utf-8'))

# net = config.CIFAR_CONV_BIG
# search_eps = [0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
# epsilons = []
# files = [f for f in listdir('./cifar_results/') if isfile(join('./cifar_results/', f))]
# for se in search_eps:
#     f_name = '{}_5_20_{:.2f}_Domain.DEEPZ.dat'.format(net, se)
#     if not f_name in files:
#         epsilons.append(se)
# if len(epsilons) > 0:
#     print(f'Evaluating {net} at {epsilons}')
#     for eps in epsilons:
#         uap_verfication_args = uap_ver.UapAnalysisArgs(
#             individual_prop_domain=Domain.DEEPZ,
#             domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
#             spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=net,                                                                                                              
#             timeout=100, output_dir='cifar_results/', radius_l=0.002, radius_r=0.25, 
#             uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#             no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
#             try_image_smoothing=False, filter_threshold=None, ligweight_diffpoly = True)
#         uap_ver.UapVerification(uap_verfication_args)
#         requests.post('https://ntfy.cmxu.io/test', data = f'Finished {net} at {eps}'.encode(encoding='utf-8'))

# uap_verfication_args = uap_ver.UapAnalysisArgs(
#     individual_prop_domain=Domain.DEEPZ,
#     domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
#     spec_type=InputSpecType.UAP_TARGETED, count=1, count_per_prop=5, eps=4.0/255, net=config.CIFAR_CONV_SMALL_DIFFAI,                                                                                                              
#     timeout=100, output_dir='cifar_results/', radius_l=0.002, radius_r=0.25, 
#     uap_mode=analyzer.UAPMode.TARGETED, compute_proportion=True, write_file=True,
#     no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
#     try_image_smoothing=False, filter_threshold=None)
# uap_ver.UapVerification(uap_verfication_args)

# net = config.CIFAR_CROWN_IBP_MEDIUM
# search_eps = [2.25, 2.5, 2.75, 3.25, 3.5, 3.75, 4.25, 4.5]
# epsilons = []
# files = [f for f in listdir('./cifar_results/') if isfile(join('./cifar_results/', f))]Pasado/Section_5_5/cls_tanh2.onnx
#             no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
#             try_image_smoothing=False, filter_threshold=None, ligweight_diffpoly = True)
#         uap_ver.UapVerification(uap_verfication_args)
#         requests.post('https://ntfy.cmxu.io/test', data = f'Finished {net} at {eps}'.encode(encoding='utf-8'))

# nets = [config.CIFAR_CONV_BIG]
# epsilons = [2.0, 2.25, 2.5, 1.75]
# for net in nets:
#     for eps in epsilons:
#         uap_verfication_args = uap_ver.UapAnalysisArgs(
#             individual_prop_domain=Domain.DEEPZ,
#             domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
#             spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=net,                                                                                                              
#             timeout=100, output_dir='cifar_results/', radius_l=0.002, radius_r=0.25, 
#             uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#             no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2,
#             try_image_smoothing=False, filter_threshold=None, ligweight_diffpoly = True)
#         uap_ver.UapVerification(uap_verfication_args)
#         requests.post('https://ntfy.cmxu.io/test', data = f'Finished {eps}'.encode(encoding='utf-8'))

# nets = [config.MNIST_CROWN_IBP]
# epsilons = [0.1, 0.105, 0.11]
# for net in nets:
#    for eps in epsilons:
#        uap_verfication_args = uap_ver.UapAnalysisArgs(
#            individual_prop_domain=Domain.DEEPPOLY,
#            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
#            spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=net,                                                                                                              
#            timeout=100, output_dir='mnist_results_new/', radius_l=0.002, radius_r=0.25, 
#            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
#            try_image_smoothing=False, filter_threshold=None)
#        uap_ver.UapVerification(uap_verfication_args)
#        requests.post('https://ntfy.cmxu.io/test', data = f'Finished {net} at {eps}'.encode(encoding='utf-8'))
#        #eps += 0.01


# eps = 4.0
# for _ in range(8):
#     uap_verfication_args = uap_ver.UapAnalysisArgs(
#         individual_prop_domain=Domain.DEEPZ,
#         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
#         spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
#         timeout=100, output_dir='par_results/', radius_l=0.002, radius_r=0.25, 
#         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
#         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
#         try_image_smoothing=False, filter_threshold=None)
#     uap_ver.UapVerification(uap_verfication_args)
#     eps += 0.5

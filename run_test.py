
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

uap_verfication_args = uap_ver.UapAnalysisArgs(
    individual_prop_domain=Domain.DEEPZ,
    domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
    spec_type=InputSpecType.UAP, count=3, count_per_prop=5, eps=float(sys.argv[2]), net=sys.argv[1],                                                                                                              
    timeout=100, output_dir='results/param_search/', radius_l=0.002, radius_r=0.25,
    uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
    no_lp_for_verified = True, debug_mode=False, track_differences=True)
thread = Thread(target=uap_ver.UapVerification, args=(uap_verfication_args,))
thread.start()
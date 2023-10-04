
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as analyzer
import src.uap_analyzer as uap_ver
from threading import Thread


class TestBasicUap(TestCase):
    # def test_mnist_uap(self):
    #     uap_verfication_args = uap_ver.UapAnalysisArgs(
    #         individual_prop_domain=Domain.DEEPZ,
    #         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
    #         spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=0.168, net=config.MNIST_BINARY,                                                                                                              
    #         timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
    #         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=False, write_file=False,
    #         no_lp_for_verified = True, debug_mode=True, track_differences=False)
    #     uap_ver.UapVerification(uap_verfication_args)

    # def test_mnist_uap_debug(self):
    #     uap_verfication_args = uap_ver.UapAnalysisArgs(
    #         individual_prop_domain=Domain.DEEPZ,
    #         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
    #         spec_type=InputSpecType.UAP, count=1, count_per_prop=2, eps=0.1, net=config.MNIST_BINARY,                                                                                                              
    #         timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
    #         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=False, write_file=False,
    #         no_lp_for_verified = True, debug_mode=True, track_differences=False)
    #     uap_ver.UapVerification(uap_verfication_args)

    # def test_mnist_uap(self):
    #     uap_verfication_args = uap_ver.UapAnalysisArgs(
    #         individual_prop_domain=Domain.DEEPZ,
    #         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
    #         spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.025, net=config.MNIST_LINEAR_50,                                                                                                              
    #         timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
    #         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
    #         no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = config.MONOTONE_PROP.RM)
    #     uap_ver.UapVerification(uap_verfication_args)
    uap_verfication_args = uap_ver.UapAnalysisArgs(
        individual_prop_domain=Domain.DEEPZ,
        domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.HOUSING, sink_label=None,
        spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=50.0, net=config.HOUSING_RM_CRIM,                                                                                                              
        timeout=100, output_dir='results/mono/', radius_l=0.002, radius_r=0.25,
        uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
        no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = config.MONOTONE_PROP.CRIM, monotone_inv=True)
    uap_ver.UapVerification(uap_verfication_args)
    # ps = [config.MONOTONE_PROP.RM, config.MONOTONE_PROP.CRIM]
    # eps = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 250.0, 500.0]
    # for p in ps:
    #     for ep in eps:
    #         uap_verfication_args = uap_ver.UapAnalysisArgs(
    #             individual_prop_domain=Domain.DEEPZ,
    #             domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.HOUSING, sink_label=None,
    #             spec_type=InputSpecType.UAP, count=98, count_per_prop=1, eps=ep, net=config.HOUSING_RM_CRIM,                                                                                                              
    #             timeout=100, output_dir='results/mono/', radius_l=0.002, radius_r=0.25,
    #             uap_mode=analyzer.UAPMode.MONOTONICITY, compute_proportion=True, write_file=True,
    #             no_lp_for_verified = True, debug_mode=False, track_differences=True, monotone_prop = p, monotone_inv=p==config.MONOTONE_PROP.CRIM)
    #         uap_ver.UapVerification(uap_verfication_args)



    # def test_cifar_uap(self): 
    #     uap_verfication_args = uap_ver.UapAnalysisArgs(
    #         individual_prop_domain=Domain.DEEPZ,
    #         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
    #         spec_type=InputSpecType.UAP, count=2, count_per_prop=5, eps=3/255, net=config.CIFAR_STANDARD_CONV,                                                                                                              
    #         timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
    #         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=False, write_file=True,
    #         no_lp_for_verified = True, debug_mode=False, track_differences=True)
    #     uap_ver.UapVerification(uap_verfication_args)

from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as analyzer
import src.uap_analyzer as uap_ver
from threading import Thread


class TestBasicUap(TestCase):   
    def test_mnist_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=4, count_per_prop=5, eps=0.005, net=config.MNIST_FFN_L2,                                                                                                              
            timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25,
            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
            try_image_smoothing=False, filter_threshold=None, fold_conv_layers=False)
        uap_ver.UapVerification(uap_verfication_args)

    def test_mnist_uap_full(self):
        eps = 0.13
        for _ in range(10):
            uap_verfication_args = uap_ver.UapAnalysisArgs(
                individual_prop_domain=Domain.DEEPPOLY,
                domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CONV_SMALL_DIFFAI,                                                                                                              
                timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25,
                uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            eps += 0.005
            uap_ver.UapVerification(uap_verfication_args)
        
        # eps = 0.13
        # for _ in range(14):
        #     uap_verfication_args = uap_ver.UapAnalysisArgs(
        #         individual_prop_domain=Domain.DEEPPOLY,
        #         domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
        #         spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps, net=config.MNIST_CROWN_IBP,                                                                                                              
        #         timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25,
        #         uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
        #         no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
        #         try_image_smoothing=False, filter_threshold=None)
        #     eps += 0.005
        #     uap_ver.UapVerification(uap_verfication_args)


    def test_mnist_uap_debug(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPPOLY,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=2, eps=0.1, net=config.MNIST_BINARY,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=False, write_file=False,
            no_lp_for_verified = True, debug_mode=True, track_differences=True)
        uap_ver.UapVerification(uap_verfication_args)

    def test_cifar_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=4.5/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
            try_image_smoothing=False, filter_threshold=None)
        uap_ver.UapVerification(uap_verfication_args)

    def test_cifar_uap_full(self):

        eps = 2.0
        for _ in range(12):
            uap_verfication_args = uap_ver.UapAnalysisArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=config.CIFAR_CROWN_IBP,                                                                                                              
                timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25, 
                uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            uap_ver.UapVerification(uap_verfication_args)
            eps += 0.35

        eps = 2.0
        for _ in range(12):
            uap_verfication_args = uap_ver.UapAnalysisArgs(
                individual_prop_domain=Domain.DEEPZ,
                domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
                spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=eps/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
                timeout=100, output_dir='results_trial/', radius_l=0.002, radius_r=0.25, 
                uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
                no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=3,
                try_image_smoothing=False, filter_threshold=None)
            uap_ver.UapVerification(uap_verfication_args)
            eps += 0.35

class TestTargetedUap(TestCase):   
    def test_mnist_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=0.1, net=config.MNIST_CONV_PGD,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=analyzer.UAPMode.TARGETED, compute_proportion=True, write_file=True,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2)
        uap_ver.UapVerification(uap_verfication_args)

    def test_mnist_uap_debug(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=0.1, net=config.MNIST_BINARY,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=analyzer.UAPMode.TARGETED, compute_proportion=False, write_file=False,
            no_lp_for_verified = True, debug_mode=True, track_differences=True)
        uap_ver.UapVerification(uap_verfication_args)

    def test_cifar_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
            spec_type=InputSpecType.UAP, count=2, count_per_prop=5, eps=3/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
            uap_mode=analyzer.UAPMode.TARGETED, compute_proportion=True, write_file=False,
            no_lp_for_verified = True, debug_mode=False, track_differences=True, lp_formulation_threshold=2)
        uap_ver.UapVerification(uap_verfication_args)

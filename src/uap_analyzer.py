import torch 
import src.specs.spec as specs
import src.config as config
from src.specs.input_spec import InputSpecType
from src.uap_analyzer_backend import UAPAnalyzerBackendWrapper
from src.uap_results import UAPResultList
from enum import Enum
from copy import deepcopy
import time


class UAPMode(Enum):
    RADIUS = 1
    VERIFICATION = 2
    TARGETED = 3
    MONOTONICITY = 4


class UapAnalysisArgs:
    def __init__(self, individual_prop_domain, domain, baseline_domain, dataset='mnist', sink_label=None, 
                 spec_type=InputSpecType.UAP, count=None, count_per_prop=2, 
                 eps=0.01, net='', timeout=30, output_dir='', radius_l=0.1, 
                 radius_r=0.3, uap_mode=UAPMode.RADIUS, cutoff_percentage = 0.5,
                 compute_proportion=False, no_lp_for_verified=False, write_file = False, 
                 debug_mode=False, track_differences=True, monotone_prop = None, monotone_inv = False, 
                 lp_formulation_threshold=2, try_image_smoothing=False, filter_threshold=None, 
                 fold_conv_layers=False, ligweight_diffpoly=False) -> None:
        self.individual_prop_domain = individual_prop_domain
        self.domain = domain
        self.baseline_domain = baseline_domain
        self.spec_type = spec_type
        self.dataset= dataset
        self.count = count
        self.count_per_prop = count_per_prop
        self.sink_label = sink_label
        self.eps = eps
        self.net = config.NET_HOME + net
        self.net_name = net
        self.timeout = timeout
        self.output_dir = output_dir
        self.radius_l = radius_l
        self.radius_r = radius_r
        self.uap_mode = uap_mode
        self.cutoff_percentage = cutoff_percentage
        self.compute_proportion = compute_proportion
        self.no_lp_for_verified = no_lp_for_verified
        self.write_file = write_file
        self.debug_mode = debug_mode
        self.track_differences = track_differences
        self.monotone_prop = monotone_prop
        self.monotone_inv = monotone_inv
        self.lp_formulation_threshold = lp_formulation_threshold
        self.try_image_smoothing = try_image_smoothing
        self.filter_threshold = filter_threshold
        # Always use all layer substitution for DiffPoly.
        self.all_layer_sub = True
        self.fold_conv_layers = fold_conv_layers
        self.lightweight_diffpoly = ligweight_diffpoly
        # if debug mode on rewrite params.
        if debug_mode == True:
            self.count = 1
            self.count_per_prop = 2
            self.eps = 6.0
            self.net_name = 'debug.net'
            self.net = 'debug.net'

def UapVerification(uap_verification_args: UapAnalysisArgs):
    start_time = time.time()
    total_local_prop_count = uap_verification_args.count * uap_verification_args.count_per_prop
    props, _ = specs.get_specs(uap_verification_args.dataset, spec_type=uap_verification_args.spec_type,
                                    count=total_local_prop_count, eps=uap_verification_args.eps, 
                                    sink_label=uap_verification_args.sink_label,
                                    debug_mode=uap_verification_args.debug_mode, 
                                    monotone_prop = uap_verification_args.monotone_prop, 
                                    monotone_inv = uap_verification_args.monotone_inv, 
                                    try_input_smoothing=uap_verification_args.try_image_smoothing,
                                    count_per_prop=uap_verification_args.count_per_prop, 
                                    net_name=uap_verification_args.net)
    if uap_verification_args.uap_mode is UAPMode.RADIUS:
        UapVerifiedRadiusBackend(props=props, uap_verification_args=uap_verification_args)
    elif uap_verification_args.uap_mode is UAPMode.VERIFICATION:
        UapVerificationBackend(props=props, uap_verification_args=uap_verification_args)
    elif uap_verification_args.uap_mode is UAPMode.TARGETED:
        UapTargetedBackend(props=props, uap_verification_args=uap_verification_args)
    elif uap_verification_args.uap_mode is UAPMode.MONOTONICITY:
        UapMonotonicityBackend(props=props, uap_verification_args=uap_verification_args)

def UapMonotonicityBackend(props, uap_verification_args):
    uap_prop_count = uap_verification_args.count
    input_per_prop = uap_verification_args.count_per_prop * 2
    uap_result_list = UAPResultList()
    for i in range(uap_prop_count):
        print(f"\n\n ***** verifying property {i} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        uap_analyzer = UAPAnalyzerBackendWrapper(props=props_to_analyze, args=uap_verification_args)
        # run the uap verification
        res = uap_analyzer.run_monotone()
        uap_result_list.add_results(res)
    if uap_verification_args.write_file == True:
       uap_result_list.analyze_monotone(uap_verification_args)

def UapTargetedBackend(props, uap_verification_args):
    uap_prop_count = uap_verification_args.count
    input_per_prop = uap_verification_args.count_per_prop
    uap_result_list = UAPResultList()
    for i in range(uap_prop_count):
        print(f"\n\n ***** verifying property {i} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        uap_analyzer = UAPAnalyzerBackendWrapper(props=props_to_analyze, args=uap_verification_args)
        # run the uap verification
        res = uap_analyzer.run_targeted()
        uap_result_list.add_results(res)
    if uap_verification_args.write_file == True:
       uap_result_list.analyze_targeted(uap_verification_args)

def UapVerifiedRadiusBackend(props, uap_verification_args):
    uap_prop_count = uap_verification_args.count
    input_per_prop = uap_verification_args.count_per_prop
    uap_result_list = UAPResultList()
    for i in range(uap_prop_count):
        print("Computing Radius\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        uap_analyzer = UAPAnalyzerBackendWrapper(props=props_to_analyze, args=uap_verification_args)
        # run the uap verification
        # Update this once it is implemented.
        # baseline_radius, uap_radius = uap_analyzer.get_radius()

def UapVerificationBackend(props, uap_verification_args):
    uap_prop_count = uap_verification_args.count
    input_per_prop = uap_verification_args.count_per_prop
    uap_result_list = UAPResultList()
    for i in range(uap_prop_count):
        print(f"\n\n ***** verifying property {i} ***** \n\n")
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop] 
        # new_prop = deepcopy(props[0])
        # new_prop.update_input(eps=0.1)       
        # props_to_analyze = [props[0], new_prop]
        uap_analyzer = UAPAnalyzerBackendWrapper(props=props_to_analyze, args=uap_verification_args)
        # run the uap verification
        res = uap_analyzer.run_timings()
        uap_result_list.add_results(res)
    if uap_verification_args.write_file == True:
       uap_result_list.analyze(uap_verification_args)

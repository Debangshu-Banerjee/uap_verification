import torch 
import src.specs.spec as specs
import src.config as config
from src.specs.input_spec import InputSpecType
from src.uap_analyzer_backend import UAPAnalyzerBackendWrapper
from src.uap_results import UAPResultList
from enum import Enum

class UAPMode(Enum):
    RADIUS = 1
    VERIFICATION = 2


class UapAnalysisArgs:
    def __init__(self, individual_prop_domain, domain, baseline_domain, dataset='mnist', sink_label=None, 
                 spec_type=InputSpecType.UAP, count=None, count_per_prop=2, 
                 eps=0.01, net='', timeout=30, output_dir='', radius_l=0.1, 
                 radius_r=0.3, uap_mode=UAPMode.RADIUS, cutoff_percentage = 0.5) -> None:
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


def UapVerification(uap_verification_args: UapAnalysisArgs):
    total_local_prop_count = uap_verification_args.count * uap_verification_args.count_per_prop
    props, _ = specs.get_specs(uap_verification_args.dataset, spec_type=uap_verification_args.spec_type,
                                    count=total_local_prop_count, eps=uap_verification_args.eps, 
                                    sink_label=uap_verification_args.sink_label)
    if uap_verification_args.uap_mode is UAPMode.RADIUS:
        UapVerifiedRadiusBackend(props=props, uap_verification_args=uap_verification_args)
    elif uap_verification_args.uap_mode is UAPMode.VERIFICATION:
        UapVerificationBackend(props=props, uap_verification_args=uap_verification_args)



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
        props_to_analyze = props[i * input_per_prop : (i+1) * input_per_prop]
        uap_analyzer = UAPAnalyzerBackendWrapper(props=props_to_analyze, args=uap_verification_args)
        # run the uap verification
        res = uap_analyzer.run()
        uap_result_list.add_results(res)
    uap_result_list.analyze(uap_verification_args)

import torch
import src.util as util 
from src.uap_results import *
from src.common import Status, Domain
from src.domains.domain_transformer import domain_transformer
from src.baseline_uap_verifier import BaselineAnalyzerBackend
from src.uap_domains.uap_domain_transformer import get_uap_domain_transformer
from src.common.network import LayerType
import time


class UAPAnalyzerBackendWrapper:
    def __init__(self, props, args) -> None:
        self.props = props
        self.args = args
        with torch.no_grad():
            self.net = util.get_net(self.args.net, self.args.dataset, debug_mode=self.args.debug_mode)    

    def get_radius(self):
        pass

    def update_props(self, eps):
        pass

    def get_opt_radius(self, radius_l, radius_r, use_uap_verifier, tolerence_lim=1e-7):
        pass
    

    def verify(self, use_uap_verifier=False):
        baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
        baseline_result = baseline_verfier.run()
        return None

    def run_monotone(self):
        start_time = time.time()
        with torch.no_grad():
            baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verfier.run()
        individual_time = time.time() - start_time
        start_time = time.time()
        uap_algorithm_res = self.run_uap_verification(domain=self.args.domain, individual_verification_results=individual_verification_results, monotone = True, monotonic_inv = self.args.monotone_inv)
        uap_time = time.time() - start_time        
        return UAPResult(baseline_res = None, UAP_res=uap_algorithm_res, individual_res=individual_verification_results, monotone = True, props = self.props, times = uap_time)


    def run_targeted(self):
        start_time = time.time()
        with torch.no_grad():
            baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verfier.run()
        individual_time = time.time() - start_time
        
        start_time = time.time()
        baseline_res = self.run_uap_verification(domain=self.args.baseline_domain, individual_verification_results=individual_verification_results, targeted = True)
        baseline_time = time.time() - start_time

        start_time = time.time()
        uap_algorithm_res = self.run_uap_verification(domain=self.args.domain, individual_verification_results=individual_verification_results, targeted = True)
        uap_time = time.time() - start_time

        return UAPResult(baseline_res=baseline_res, UAP_res=uap_algorithm_res, individual_res=individual_verification_results, targeted = True, times = [individual_time, baseline_time, uap_time, sum([res.constraint_time for res in uap_algorithm_res]), sum([res.optimize_time for res in uap_algorithm_res])], props = self.props)


    def run_timings(self):
        start_time = time.time()
        with torch.no_grad():
            baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verfier.run()
        individual_time = time.time() - start_time
        # individual_verification_results.timings = LP_TIMINGS(total_time=individual_time,
        #                                                     constraint_formulation_time=None,
        #                                                     optimization_time=None)

        start_time = time.time()
        baseline_res = self.run_uap_verification(domain=self.args.baseline_domain, 
                                                individual_verification_results=individual_verification_results, diff = False)
        baseline_time = time.time() - start_time
        baseline_res.timings = LP_TIMINGS(total_time=individual_time + baseline_time,
                                                            constraint_formulation_time=None,
                                                            optimization_time=None)
        
        # Run the uap verifier without diff constraints.
        start_time = time.time()
        uap_algorithm_no_diff_res = self.run_uap_verification(domain=self.args.domain, 
                                                 individual_verification_results=individual_verification_results, 
                                                 diff=False)
        uap_diff_time = time.time() - start_time

        # Populate timings.
        uap_algorithm_no_diff_res.timings = LP_TIMINGS(total_time=(individual_time + uap_diff_time), 
                                     constraint_formulation_time=uap_algorithm_no_diff_res.constraint_time,
                                     optimization_time=uap_algorithm_no_diff_res.optimize_time)
        # Run the uap verifier with diff constraints.
        uap_algorithm_res = None
        uap_timing = None
        if self.args.track_differences is True:
            start_time = time.time()
            uap_algorithm_res = self.run_uap_verification(domain=self.args.domain, 
                                                    individual_verification_results=individual_verification_results, 
                                                    diff=True)
            uap_time = time.time() - start_time
            uap_algorithm_res.timings = LP_TIMINGS(total_time=(individual_time + uap_time), 
                                constraint_formulation_time=uap_algorithm_res.constraint_time,
                                optimization_time=uap_algorithm_res.optimize_time)
        
        return UAPResult(baseline_res=baseline_res, UAP_res=uap_algorithm_res, individual_res=individual_verification_results,
                          result_with_no_diff=uap_algorithm_no_diff_res, times=None, individual_time=individual_time)

    def run(self) -> UAPResult:
        # Baseline results correspond to running each property individually.
        with torch.no_grad():
            baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verfier.run()
        baseline_res = self.run_uap_verification(domain=self.args.baseline_domain, 
                                                 individual_verification_results=individual_verification_results)
        uap_algorithm_res = self.run_uap_verification(domain=self.args.domain, 
                                                 individual_verification_results=individual_verification_results)
        return UAPResult(baseline_res=baseline_res, UAP_res=uap_algorithm_res)


    def run_uap_verification(self, domain, individual_verification_results, targeted = False, monotone = False, monotonic_inv = False, diff = True):
        uap_verifier = get_uap_domain_transformer(domain=domain, net=self.net, props=self.props, 
                                                           args=self.args, 
                                                           baseline_results=individual_verification_results)
        if self.args.no_lp_for_verified == True and domain == Domain.UAP_DIFF:
            uap_verifier.no_lp_for_verified = True
        return uap_verifier.run(proportion=self.args.compute_proportion, targeted = targeted, monotone = monotone, monotonic_inv = monotonic_inv, diff = diff)

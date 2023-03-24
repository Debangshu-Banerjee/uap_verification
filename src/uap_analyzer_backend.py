import torch
import src.util as util 
from src.uap_results import *
from src.common import Status
from src.domains.domain_transformer import domain_transformer
import time


class UAPAnalyzerBackendWrapper:
    def __init__(self, props, args) -> None:
        self.props = props
        self.args = args
        self.net = util.get_net(self.args.net, self.args.dataset)
        self.prop_coefs = []
        self.prop_centers = []
    
    def run(self) -> UAPResult:
        baseline_verfier = UAPBaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
        baseline_result = baseline_verfier.run()
        if baseline_result.status is Status.VERIFIED:
            UAP_result = baseline_result
        else:
            print("Baseline lb", baseline_result.global_lb)
            coefs, centers =  baseline_verfier.get_centers_coefs()
            UAP_result = UAPAnalyzerBackend(props=self.props, net=self.net, args=self.args, coefs=coefs, centers=centers).run()
        return UAPResult(UAP_res=UAP_result, baseline_res=baseline_result)


class UAPAnalyzerBackend:
    def __init__(self, props, net, args, coefs=None, centers=None):
        self.props = props
        self.net = net 
        self.args = args
        self.coefs = coefs
        self.centers = centers

    def run(self) -> UAPSingleRes:
        start_time = time.time()
        problem_status = Status.UNKNOWN
        if self.coefs is None or self.centers is None:
            raise ValueError("Coefs and centers must be both not null") 
        with torch.no_grad():
            transformer = domain_transformer(net=self.net, prop=self.props[0].get_input_clause(0), 
                                         domain=self.args.domain)
            transformer.set_coefs_centers(coefs=self.coefs, centers=self.centers)
            lb = transformer.compute_lb()
        print("UAP verifier lower bound", lb)
        if lb >= 0:
            problem_status = Status.VERIFIED
        time_taken = time.time() - start_time
        result = UAPSingleRes(domain=self.args.domain, 
                                input_per_prop=self.args.count_per_prop, status=problem_status, 
                                global_lb=lb, time_taken=time_taken)
        return result

class UAPBaselineAnalyzerBackend:
    def __init__(self, props, net, args):
        self.props = props
        self.net = net 
        self.args = args
        self.centers = []
        self.coefs = []

    def get_centers_coefs(self):
        return self.coefs, self.centers

    def run(self) -> UAPSingleRes:
        start_time = time.time()
        baseline_status = Status.UNKNOWN
        global_lb = None
        for prop in self.props:
            assert prop.get_input_clause_count() == 1
            transformer = domain_transformer(net=self.net, prop=prop.get_input_clause(0), domain=self.args.baseline_domain)
            lb = transformer.compute_lb()
            min_lb = torch.min(lb)
            if global_lb is None or global_lb < min_lb:
                global_lb = min_lb 
            if min_lb >= 0:
                baseline_status = Status.VERIFIED
                break
            if hasattr(transformer, 'final_coef_center'):
                coef, center = transformer.final_coef_center()
                self.centers.append(center)
                self.coefs.append(coef)
        time_taken = time.time() - start_time
        baseline_result = UAPSingleRes(domain=self.args.baseline_domain, 
                                       input_per_prop=self.args.count_per_prop, status=baseline_status, 
                                       global_lb=global_lb, time_taken=time_taken)
        return baseline_result
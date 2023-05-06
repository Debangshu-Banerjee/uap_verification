import torch
from src.uap_results import UAPSingleRes
from src.domains.diff_deeppoly import DiffDeepPoly

class UapDiff:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props
        self.args = args
        self.baseline_results = baseline_results
        self.difference_lbs_dict = {}
        self.difference_ubs_dict = {}
        self.input_list = []

    def compute_difference_dict(self):
        for i in range(len(self.baseline_results)):
            for j in range(i+1, len(self.baseline_results)):
                result1 = self.baseline_results[i]
                result2 = self.baseline_results[j]
                input1 = result1.input
                input1_lbs = result1.layer_lbs
                input1_ubs = result1.layer_ubs
                input2 = result2.input
                input2_lbs = result2.layer_lbs
                input2_ubs = result2.layer_ubs
                diff_poly_ver = DiffDeepPoly(input1=input1, input2=input2, net=self.net, 
                             lb_input1=input1_lbs, ub_input1=input1_ubs,
                             lb_input2=input2_lbs, ub_input2=input2_ubs, device='cpu')
                delta_lbs, delta_ubs = diff_poly_ver.run()
                self.difference_lbs_dict[(i, j)] = delta_lbs
                self.difference_ubs_dict[(i, j)] = delta_ubs
            self.input_list.append(self.baseline_results[i].input)        

    def run(self) -> UAPSingleRes:
        print("Started differential verification")
        self.compute_difference_dict()
        print("Differential verification completed")
        # Call the lp formulation with the differential lp code.
        return None

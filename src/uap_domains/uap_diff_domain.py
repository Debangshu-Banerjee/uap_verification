import torch
from src.uap_results import UAPSingleRes
from src.domains.diff_deeppoly import DiffDeepPoly
from src.uap_lp_new import UAPLPtransformer
import time
from src.common import Status

class UapDiff:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props
        self.args = args
        self.total_props = len(self.props)
        self.baseline_results = baseline_results
        self.difference_lbs_dict = {}
        self.difference_ubs_dict = {}
        self.input_list = []
        self.eps = args.eps
        self.input_lbs = []
        self.input_ubs = []
        self.constr_matrices = []
        self.last_conv_diff_structs = []
        self.no_lp_for_verified = self.args.no_lp_for_verified
        self.lp_formulation_threshold = self.args.lp_formulation_threshold
        self.baseline_verified_props = 0
        self.noise_ind = baseline_results[0].noise_ind
        self.monotone_lp = False
        #self.eps = baseline_results[0].eps

    def compute_difference_dict(self, monotone = False):
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
                with torch.no_grad():
                    diff_poly_ver = DiffDeepPoly(input1=input1, input2=input2, net=self.net, 
                                lb_input1=input1_lbs, ub_input1=input1_ubs,
                                lb_input2=input2_lbs, ub_input2=input2_ubs, device='cpu', 
                                noise_ind = self.noise_ind, eps = self.eps, 
                                monotone = monotone, monotone_prop = self.args.monotone_prop, use_all_layers=self.args.all_layer_sub, 
                                lightweight_diffpoly=self.args.lightweight_diffpoly)
                    delta_lbs, delta_ubs = diff_poly_ver.run()
                self.difference_lbs_dict[(i, j)] = delta_lbs
                self.difference_ubs_dict[(i, j)] = delta_ubs

    def populate_input_list(self):
        for baseline_res in self.baseline_results:
            self.input_list.append(baseline_res.input)

    def populate_lbs_and_ubs(self):
        for i in range(len(self.baseline_results)):
            self.input_lbs.append(self.baseline_results[i].layer_lbs)
            self.input_ubs.append(self.baseline_results[i].layer_ubs)

    def populate_matrices(self):
        for prop in self.props:
            self.constr_matrices.append(prop.get_input_clause(0).output_constr_mat())
    
    def populate_diff_structs(self):
        for res in self.baseline_results:
            self.last_conv_diff_structs.append(res.last_conv_diff_struct)
    
    def get_negative_threshold(self):
        lb_list = []
        for i, prop in enumerate(self.props):
            if torch.min(self.baseline_results[i].final_lb) < 0.0:
                lb_list.append(torch.min(self.baseline_results[i].final_lb))
        lb_list.sort()
        if len(lb_list) <= self.lp_formulation_threshold:
            return -1e9
        else:
            return lb_list[(-self.lp_formulation_threshold - 1)]

    def prune_verified_props(self):
        new_props = []
        new_baseline_results = []
        threshold = self.get_negative_threshold()
        if self.args.filter_threshold is not None:
            threshold = max(threshold, self.args.filter_threshold)
        for i, prop in enumerate(self.props):
            lb = torch.min(self.baseline_results[i].final_lb)
            if  lb >= 0.0 or lb <= threshold:
                if lb >= 0.0:
                    self.baseline_verified_props += 1
                continue
            new_props.append(prop)
            new_baseline_results.append(self.baseline_results[i])
        self.props = new_props
        self.baseline_results = new_baseline_results

    def run(self, proportion=False, targeted = False, monotone = False, monotonic_inv = False, diff = True) -> UAPSingleRes:
        start_time = time.time()
        if self.no_lp_for_verified == True and not targeted and not monotone:
            self.prune_verified_props()
        # Do not invoke diffPoly if the diff constraints is disabled.
        if diff:
            self.compute_difference_dict(monotone = monotone)
        self.populate_input_list()
        self.populate_lbs_and_ubs()
        if self.args is not None and self.args.fold_conv_layers is True:
            self.populate_diff_structs()

        if monotone and not self.monotone_lp:
            verified_status = Status.UNKNOWN
            print(self.difference_lbs_dict[(0,1)][-1], self.difference_ubs_dict[(0,1)][-1])
            if not monotonic_inv:
                if self.difference_lbs_dict[(0,1)][-1] >= 0:
                    verified_status = Status.VERIFIED
            else:
                if self.difference_ubs_dict[(0,1)][-1] <= 0:
                    verified_status = Status.VERIFIED
            return UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=None, time_taken=None, 
                    verified_proportion=None) 

        self.populate_matrices()
        if self.args.debug_mode is True:
            print(self.input_list)
            print("input1 ", self.input_list[0])
            print("input2 ", self.input_list[1])
            print(self.input_lbs)
            print(self.input_ubs)
            print(self.difference_lbs_dict)
            print(self.difference_ubs_dict)
            print(self.eps)
            print(self.net[0].weight, self.net[0].bias)
            print(self.net[1].weight, self.net[1].bias)
            print(self.net[2].weight, self.net[2].bias)
            
        # Call the lp formulation with the differential lp code.
        if not diff:
            uap_lp_transformer = UAPLPtransformer(mdl=self.net, xs=self.input_list, 
                                                eps=self.eps, x_lbs=self.input_lbs,
                                                x_ubs=self.input_ubs, d_lbs=self.difference_lbs_dict,
                                                d_ubs=self.difference_ubs_dict, 
                                                constraint_matrices=self.constr_matrices,
                                                last_conv_diff_structs=self.last_conv_diff_structs,
                                                debug_mode=self.args.debug_mode,
                                                track_differences=False, props = self.props, monotone = monotone, args=self.args)
        else:
            uap_lp_transformer = UAPLPtransformer(mdl=self.net, xs=self.input_list, 
                                                eps=self.eps, x_lbs=self.input_lbs,
                                                x_ubs=self.input_ubs, d_lbs=self.difference_lbs_dict,
                                                d_ubs=self.difference_ubs_dict, constraint_matrices=self.constr_matrices,
                                                last_conv_diff_structs=self.last_conv_diff_structs,
                                                debug_mode=self.args.debug_mode,
                                                track_differences=self.args.track_differences, props = self.props, 
                                                monotone = monotone, args=self.args)
        
        # Formulate the Lp problem.
        lp_start_time = time.time()
        uap_lp_transformer.create_lp()
       
        verified_percentages = None
        global_lb = None
        verified_status = Status.UNKNOWN
        if monotone:
            verified_percentages = uap_lp_transformer.optimize_monotone(monotone)
            print(verified_percentages)
            if verified_percentages >= 0:
                verified_status = Status.VERIFIED
            return UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                status=verified_status, global_lb=None, time_taken=None, 
                verified_proportion=None) 
        elif targeted:
            print('hi')
            verified_percentages, bin_sizes = uap_lp_transformer.optimize_targeted()
            print("Diff global proportion ", verified_percentages)
            results = []
            for i, an in enumerate(verified_percentages):
                verified_proportion = an
                if verified_proportion >= self.args.cutoff_percentage:
                    verified_status = Status.VERIFIED
                results.append(UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                        status=verified_status, global_lb=global_lb, time_taken=None, 
                        verified_proportion=verified_proportion, bin_size = bin_sizes[i], constraint_time = uap_lp_transformer.constraint_time, optimize_time = uap_lp_transformer.optimize_time))
                verified_status = Status.UNKNOWN
            return results
        else:
            if proportion == False:
                global_lb = uap_lp_transformer.optimize_lp()
                print("Diff global lb", global_lb)
                if global_lb >= 0.0:
                    verified_status = Status.VERIFIED            
            else:
                verified_percentages = uap_lp_transformer.optimize_milp_percent()
                verified_props = verified_percentages * len(self.props)
                verified_percentages = (verified_props + self.baseline_verified_props) / self.total_props
                print("Diff Verified percentages", verified_percentages)
                if verified_percentages >= self.args.cutoff_percentage:
                    verified_status = Status.VERIFIED

        time_taken = time.time() - start_time
        return UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=None, time_taken=time_taken, 
                    verified_proportion=verified_percentages, constraint_time = uap_lp_transformer.constraint_time, optimize_time = uap_lp_transformer.optimize_time)

from src.common import Status
import torch

class LP_TIMINGS:
    def __init__(self, total_time, constraint_formulation_time, optimization_time):
        self.total_time = total_time
        self.constraint_formulation_time = constraint_formulation_time
        self.optimization_time = optimization_time

class UAPSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, 
                 time_taken, verified_proportion, constraint_time = None, 
                 optimize_time = None, bin_size = None, timings=None):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken
        self.verified_proportion = verified_proportion
        self.constraint_time = constraint_time
        self.optimize_time = optimize_time
        self.bin_size = bin_size
        # Holds the final timings of the entire analysis.
        self.timings = timings

    def print(self):
        print("Domain ", self.domain)
        print("Time taken ", self.time_taken)
        print("Global lb ", self.global_lb)
        print("Status ", self.status)
        print("Input per prop ", self.input_per_prop)
        print("Verified proportion", self.verified_proportion)

class UAPResult:
    def __init__(self, UAP_res, baseline_res, individual_res = None, result_with_no_diff=None,
                targeted = False, times = None, props = None, monotone = False, individual_time=None):
        self.baseline_res = baseline_res
        self.individual_time = individual_time
        self.UAP_res = UAP_res
        self.result_with_no_diff = result_with_no_diff
        self.individual_res = individual_res
        self.times = times
        self.targeted = targeted
        self.props = props
        self.monotone = monotone

class UAPResultList:
    def __init__(self) -> None:
        self.result_list = []

    def add_results(self, res: UAPResult):
        self.result_list.append(res)

    def analyze(self, args):
        count = args.count
        individual_verified_count = 0
        baseline_verified_count = 0
        uap_verified_without_diff = 0
        uap_verified_count = 0
        times = [0, 0, 0, 0]
        layerwise_constraint_time = 0
        layerwise_optimization_time = 0
        diff_constraint_time = 0
        diff_optimization_time = 0
        filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}_{args.individual_prop_domain}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            individual_res = res.individual_res
            baseline_res = res.baseline_res
            uap_no_diff_res = res.result_with_no_diff
            UAP_res = res.UAP_res
            if individual_res is not None:
                veri = sum([torch.min(res.final_lb) >= 0 for res in individual_res])
                individual_verified_count += veri
                if res.individual_time is not None:
                    times[0] += res.individual_time
            if baseline_res is not None and baseline_res.verified_proportion is not None:
                baseline_verified_count += baseline_res.verified_proportion * args.count_per_prop
                if times[1] is not None and baseline_res.timings is not None:
                    times[1] += baseline_res.timings.total_time
            if uap_no_diff_res is not None and uap_no_diff_res.verified_proportion is not None:
                uap_verified_without_diff += uap_no_diff_res.verified_proportion * args.count_per_prop
                if times[2] is not None and uap_no_diff_res.timings is not None:
                    times[2] += uap_no_diff_res.timings.total_time
                    if uap_no_diff_res.timings.constraint_formulation_time is not None:
                        layerwise_constraint_time += uap_no_diff_res.timings.constraint_formulation_time
                    if uap_no_diff_res.timings.optimization_time is not None:
                        layerwise_optimization_time += uap_no_diff_res.timings.optimization_time
            if UAP_res is not None and UAP_res.verified_proportion is not None:
                uap_verified_count += UAP_res.verified_proportion * args.count_per_prop
                if times[3] is not None and UAP_res.timings is not None:
                    times[3] += UAP_res.timings.total_time
                    if UAP_res.timings.constraint_formulation_time is not None:
                        diff_constraint_time += UAP_res.timings.constraint_formulation_time
                    if UAP_res.timings.optimization_time is not None:
                        diff_optimization_time += UAP_res.timings.optimization_time
        print(f'count {count}')
        for i, _  in enumerate(times):
            print(f'time {times[i]}')
            if times[i] is not None and count > 0:
                times[i] /= count
            print(f'time {times[i]}')
        if count > 0:
            layerwise_constraint_time /= count
            layerwise_optimization_time /= count
            diff_constraint_time /= count
            diff_optimization_time /= count
    
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Individual verified: {individual_verified_count}\n')
        file.write(f'Baseline verified: {baseline_verified_count}\n')
        file.write(f'Uap no diff verified {uap_verified_without_diff if uap_no_diff_res is not None and uap_no_diff_res.verified_proportion is not None else "x"}\n')
        file.write(f'Uap verified: {uap_verified_count}\n')
        file.write(f'Extra verified: {uap_verified_count - baseline_verified_count}\n')
        file.write(f'Extra diff verified {uap_verified_count - uap_verified_without_diff}\n')
        file.write(f'Avg. times {times}\n')

        # Write the formulation and optimization times.
        file.write('\n\n\n')
        if uap_no_diff_res is not None and uap_no_diff_res.timings is not None:
            file.write(f'No diff constraint time {layerwise_constraint_time}\n')
            file.write(f'No diff optimization time {layerwise_optimization_time}\n')

        if UAP_res is not None and UAP_res.timings is not None:
            file.write(f'With diff constraint time {diff_constraint_time}\n')
            file.write(f'With diff optimization time {diff_optimization_time}\n')

        file.close()

    def analyze_monotone(self, args):
        diff_verified_count = 0
        # lp_verified_count = 0
        filename = args.output_dir + f'{args.net_name}_{args.monotone_prop}_{args.count_per_prop}_{args.count}_{args.eps}.dat'
        file = open(filename, 'a+')
        times = 0
        for i, res in enumerate(self.result_list):
            #file.write(f'\nProperty No. {i}\n\n')
            UAP_res = res.UAP_res
            if UAP_res.status == Status.VERIFIED:
                diff_verified_count += 1
            times += res.times
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Diff verified: {diff_verified_count}\n')
        file.write(f'Time: {times}')
        # file.write(f'LP verified: {lp_verified_count}\n')
        file.close()  
        
    def analyze_targeted(self, args):
        count = args.count
        individual_verified_count = torch.zeros(10)
        baseline_verified_count = torch.zeros(10)
        uap_verified_count = torch.zeros(10)
        times = [0, 0, 0, 0, 0]
        diff = ''
        if args.track_differences is True:
            diff = '_diff'        
        filename = args.output_dir + f'target_{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}{diff}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            individual_res = res.individual_res
            baseline_res = res.baseline_res
            UAP_res = res.UAP_res
            file.write(f'\nProperty No. {i}\n\n')
            if individual_res is not None:
                #print("DeepZ lbs", [res.final_lb for res in self.baseline_results])
                deepz_res = [[] for i in range(10)]
                for i in range(len(individual_res)):
                    for j in range(10):
                        if res.props[i].out_constr.label == j:
                            continue
                        deepz_res[j].append((individual_res[i].target_ubs[j]).min() <= 0)
                veri = torch.tensor([(sum(res)).item() for res in deepz_res])
                individual_verified_count += veri
                file.write(f"individual verified proportion {[(veri[i]/len(deepz_res[i])).item() for i in range(len(deepz_res))]}\n")
            if baseline_res[0].verified_proportion is not None:
                baseline_verified_count += torch.tensor([base_res.verified_proportion * base_res.bin_size for base_res in baseline_res])
                #baseline_verified_count += baseline_res.verified_proportion * args.count_per_prop
                file.write(f"baseline verified proportion {[base_res.verified_proportion for base_res in baseline_res]}\n")
            if UAP_res[0].verified_proportion is not None:
                uap_verified_count += torch.tensor([uap_res.verified_proportion * uap_res.bin_size for uap_res in UAP_res])
                file.write(f"Uap verified proportion {[uap_res.verified_proportion for uap_res in UAP_res]}\n")
            if res.times is not None:
                times[0] += res.times[0]
                times[1] += res.times[0] + res.times[1]
                times[2] += res.times[0] + res.times[2]
                times[3] += 0 if res.times[3] is None else res.times[3]
                times[4] += 0 if res.times[4] is None else res.times[4]
                file.write(f'Times: {res.times}\n')
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Individual verified: {individual_verified_count.tolist()} total: {sum(individual_verified_count).item()}\n')
        file.write(f'Baseline verified: {baseline_verified_count.tolist()} total: {sum(baseline_verified_count).item()}\n')
        file.write(f'Uap verified: {uap_verified_count.tolist()} total: {sum(uap_verified_count).item()}\n')
        file.write(f'Extra verified: {(uap_verified_count - baseline_verified_count).tolist()} total: {sum(uap_verified_count).item() - sum(baseline_verified_count).item()}\n')
        file.write(f'times: {times}\n')
        file.close()                       
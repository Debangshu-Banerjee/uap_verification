from src.common import Status
import torch

class UAPSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, time_taken, verified_proportion, constraint_time = None, optimize_time = None, bin_size = None):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken
        self.verified_proportion = verified_proportion
        self.constraint_time = constraint_time
        self.optimize_time = optimize_time
        self.bin_size = bin_size

    def print(self):
        print("Domain ", self.domain)
        print("Time taken ", self.time_taken)
        print("Global lb ", self.global_lb)
        print("Status ", self.status)
        print("Input per prop ", self.input_per_prop)
        print("Verified proportion", self.verified_proportion)

class UAPResult:
    def __init__(self, UAP_res, baseline_res, individual_res = None, targeted = False, times = None, props = None, monotone = False):
        self.baseline_res = baseline_res
        self.UAP_res = UAP_res
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
        uap_verified_count = 0
        times = [0, 0, 0, 0, 0]        
        filename = args.output_dir + f'{args.net_name}.dat'#_{args.count_per_prop}_{args.count}_{args.eps}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            individual_res = res.individual_res
            baseline_res = res.baseline_res
            UAP_res = res.UAP_res
            #file.write(f'\nProperty No. {i}\n\n')
            if individual_res is not None:
                veri = sum([torch.min(res.final_lb) >= 0 for res in individual_res])
                individual_verified_count += veri
                #file.write(f"individual verified proportion {veri/args.count_per_prop}\n")
            if baseline_res.verified_proportion is not None:
                baseline_verified_count += baseline_res.verified_proportion * args.count_per_prop
                #file.write(f"baseline verified proportion {baseline_res.verified_proportion}\n")
            if UAP_res.verified_proportion is not None:
                uap_verified_count += UAP_res.verified_proportion * args.count_per_prop
                #file.write(f"Uap verified proportion {UAP_res.verified_proportion}\n")
            if res.times is not None:
                times[0] += res.times[0]
                times[1] += res.times[0] + res.times[1]
                times[2] += res.times[0] + res.times[2]
                times[3] += 0 if res.times[3] is None else res.times[3]
                times[4] += 0 if res.times[4] is None else res.times[4]
                file.write(f'Times: {res.times}\n')
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Individual verified: {individual_verified_count}\n')
        file.write(f'Baseline verified: {baseline_verified_count}\n')
        file.write(f'Uap verified: {uap_verified_count}\n')
        file.write(f'Extra verified: {uap_verified_count - baseline_verified_count}\n')
        file.write(f'times: {times}\n')
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
            # if args.monotone_inv:
            #     lp_verified_count += 1 if UAP_res.verified_proportion >= 0 else 0
            # else:
            #     lp_verified_count += 1 if UAP_res.verified_proportion <= 0 else 0
            #file.write(f"diff verified: {UAP_res.verified_proportion >= 0}\n")
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
        filename = args.output_dir + f'target_{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}.dat'
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
from src.common import Status

class UAPSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, time_taken):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken

    def print(self):
        print("Domain ", self.domain)
        print("Time taken ", self.time_taken)
        print("Global lb ", self.global_lb)
        print("Status ", self.status)
        print("Input per prop ", self.input_per_prop)

class UAPResult:
    def __init__(self, UAP_res, baseline_res):
        self.baseline_res = baseline_res
        self.UAP_res = UAP_res

class UAPResultList:
    def __init__(self) -> None:
        self.result_list = []

    def add_results(self, res: UAPResult):
        self.result_list.append(res)

    def analyze(self, args):
        count = args.count
        baseline_verified_count = 0
        uap_verified_count = 0        
        total_baseline_time = 0
        total_additional_time = 0
        for res in self.result_list:
            baseline_res = res.baseline_res
            UAP_res = res.UAP_res
            total_baseline_time += baseline_res.time_taken            
            if baseline_res.status is Status.VERIFIED:
                baseline_verified_count += 1
            else:
                total_additional_time += UAP_res.time_taken
            if UAP_res.status is Status.VERIFIED:
                uap_verified_count += 1
        avg_baseline_time = total_baseline_time / count
        avg_uap_time = (total_baseline_time + total_additional_time) / count
        baseline_accuracy = baseline_verified_count / count * 100
        uap_accuracy = uap_verified_count / count * 100
        filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}.dat'
        file = open(filename, 'a+') 
        file.write(f'Eps : {args.eps}\n')
        file.write(f'Baseline-accuray : {baseline_accuracy}\n')
        file.write(f'Uap-ver-accuray : {uap_accuracy}\n')
        file.write(f'Avg-baseline-time : {avg_baseline_time}\n')
        file.write(f'Avg-uap-ver-time : {avg_uap_time}\n')
        file.close()                 
import matplotlib.pyplot as plt
from copy import copy
if __name__ == '__main__':
    net_names = [
        # 'mnist_0.1.onnx',
        # 'mnist_cnn_2layer_width_1_best.pth',
        'mnistconvSmallRELU__PGDK.onnx',
        'mnist-net_256x2.onnx',
        ]
    count_per_prop = 5
    count = 50
    result_dir = './'
    for net_name in net_names:
        filename = result_dir + f'{net_name}_{count_per_prop}_{count}.dat'
        file = open(filename, 'r')
        line = file.readline()
        baseline_time = {}
        uap_time = {}
        baseline_accuracy = {}
        uap_accuracy = {}
        eps = None
        while line:
            tokens = line.split(' ', 3)
            if tokens[0] == 'Eps':
                eps = float(tokens[2])
            if tokens[0] == 'Baseline-accuray':
                assert eps is not None
                baseline_accuracy[eps] = float(tokens[2])          
            if tokens[0] == 'Uap-ver-accuray':
                assert eps is not None
                uap_accuracy[eps] = float(tokens[2])
            if tokens[0] == 'Avg-baseline-time':
                assert eps is not None
                baseline_time[eps] = float(tokens[2])
            if tokens[0] == 'Avg-uap-ver-time':
                assert eps is not None
                uap_time[eps] = float(tokens[2])
            line = file.readline()
        eps_list = []
        for x in baseline_accuracy.keys():
            eps_list.append(x)
        eps_list.sort()
        time_baseline = []
        time_uap = []
        accuracy_baseline = []
        accuracy_uap = []
        for eps in eps_list:
            time_baseline.append(baseline_time[eps])
            time_uap.append(uap_time[eps])
            accuracy_baseline.append(baseline_accuracy[eps])
            accuracy_uap.append(uap_accuracy[eps])
        plt.plot(eps_list, time_baseline, 'r-o', label ='baseline')
        plt.plot(eps_list, time_uap, 'b-o', label='uap verifier')
        plt.xlabel("Epsilon")
        plt.ylabel("Avg. time (sec)")
        plt.legend()
        plt.title('Verification Time')
        plt.savefig(filename+'_time.png')
        
        plt.clf()
        plt.plot(eps_list, accuracy_baseline, 'r-o', label ='baseline')
        plt.plot(eps_list, accuracy_uap, 'b-o', label='uap verifier')
        plt.yticks([i * 10 for i in range(11)])
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy %")
        plt.legend()
        plt.title('Verification Accuracy')
        plt.savefig(filename+'_accuracy.png')
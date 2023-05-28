from src.specs.input_spec import InputSpecType

NET_HOME = "src/nets/"
DEVICE = 'cpu'


class Args:
    def __init__(self, net, domain, count=None, eps=0.01, dataset='mnist', spec_type=InputSpecType.LINF, split=None,
                 pt_method=None, timeout=None, parallel=False, initial_split=False, attack=None):
        self.net = NET_HOME + net
        self.domain = domain
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.spec_type = spec_type
        self.split = split
        self.pt_method = pt_method
        self.timeout = timeout
        self.parallel = parallel
        self.initial_split = initial_split
        self.attack = attack
        self.ignore_properties = []


class PruningArgs:
    def __init__(self, desried_perturbation=None, layers_to_prune=None, swap_layers= False, node_wise_bounds=False,
                        unstructured_pruning=True, structured_pruning=False, maximum_iteration=10, accuracy_drop=None):
        self.desired_perturbation= desried_perturbation
        self.layers_to_prune = layers_to_prune
        self.swap_layers = swap_layers
        self.node_wise_bounds= node_wise_bounds
        self.unstructured_pruning = unstructured_pruning
        self.structured_pruning = structured_pruning
        self.maximum_iteration = maximum_iteration
        self.accuracy_drop = accuracy_drop

log_file = "log.txt"
log_enabled = False


def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "IVAN"
baseline = "Baseline"

# Networks
MNIST_FFN_01 = "mnist_0.1.onnx"  # 9508 neuron
MNIST_FFN_03 = "mnist_0.3.onnx"
MNIST_FFN_DIFFAI = "mnistconvSmallRELUDiffAI.onnx"
MNIST_FFN_PGD = "mnistconvSmallRELU__PGDK.onnx"
MNIST_CONV_PGD_03 = "convMedGRELU__PGDK_w_0.3.onnx"
MNIST_FFN_L2 = "mnist-net_256x2.onnx"  # 512 neurons
MNIST_FFN_L4 = "mnist-net_256x4.onnx"   # 1024 neurons
MNIST_FFN_torch1 = "cpt/fc1.pt"
MNIST_STANDARD_MODIFIED = "mnist_standard_modified.pt"
MNIST_LINEAR_50 = "mnist_relu_3_50.onnx"
MNIST_LINEAR_100 = "mnist_relu_3_100.onnx"

# MNIST crown-ibp network
MNIST_CROWN_IBP = "mnist_cnn_2layer_width_1_best.pth"
MNIST_CROWN_IBP_MODIFIED = "mnist_cnn_2layer_width_1_best_modified.pth"

# CIFAR Common Network
CIFAR_STANDARD_4 = "cifar_relu_4_100.onnx"
CIFAR_STANDARD_6 = "ffnnRELU__Point_6_500.onnx"
CIFAR_STANDARD_CONV = "convSmallRELU__Point.onnx"

# CIFAR crown-ibp Network
CIFAR_CROWN_IBP = "cifar_cnn_2layer_width_2_best.pth"

# CIFAR Networks
CIFAR_CONV_2_255 = "cifar10_2_255.onnx"  # 49402 neurons
CIFAR_CONV_8_255 = "cifar10_8_255.onnx"  # 16634 neurons
CIFAR_CONV_SMALL = "convSmall_pgd_cifar.onnx"   # 3,604 neurons
CIFAR_CONV_COLT = "cifar10_8_255_colt.onnx"
CIFAR_CONV_DIFFAI = "cifar10convSmallRELUDiffAI.onnx"
CIFAR_CONV_BIG = "convBigRELU__DiffAI_cifar.onnx"  # 62,464 neurons

# OVAL21 CIFAR
CIFAR_OVAL_BASE = "oval21/cifar_base_kw.onnx"   # 3172 neurons
CIFAR_OVAL_WIDE = "oval21/cifar_wide_kw.onnx"   # 6244 neurons
CIFAR_OVAL_DEEP = "oval21/cifar_deep_kw.onnx"   # 6756 neurons


def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name

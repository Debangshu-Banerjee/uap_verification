import torch
import torch.nn.functional as F
from src.common.network import LayerType
from src.util import compute_input_shapes


class DeepPolyStruct:
    def __init__(self, lb_coef, lb_bias, ub_coef, ub_bias) -> None:
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
        self.ub_coef = ub_coef
        self.ub_bias = ub_bias
    
    def populate(self, lb_coef, lb_bias, ub_coef, ub_bias):
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
        self.ub_coef = ub_coef
        self.ub_bias = ub_bias


class DeepPolyTransformerOptimized:
    def __init__(self, prop, device=''):
        self.lbs = []
        self.ubs = []
        self.prop = prop
        self.layers = []
        # keep track of the final lb coef and bias
        # this will be used in baseline LP formulation.
        self.final_lb_coef = None
        self.final_lb_bias = None
        # Track input bounds and input size.
        self.size = prop.get_input_size()
        self.prop = prop
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.input_shape = None
        # Tracking shpes for supporting conv layers.
        self.shapes = []
        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)
        elif self.size == 2:
            # For debug network
            self.shape = (1, 1, 2)
        self.shapes.append(self.shape)

        self.device = device

    
    def concrete_substitution(self, diff_struct):
        lb = 
    
    def analyze_linear(self, linear_wt, bias, diff_struct):
        pass
    
    def analyze_conv(self, diff_struct):
        pass

    def analyze_relu(self, diff_struct):
        pass 

    def get_layer_size(self, layer_index):
        layer = self.layers[layer_index]
        if layer.type not in [LayerType.Linear, LayerType.Conv2D]:
            raise ValueError("Should only be called for linear or conv layers")
        if layer.type is LayerType.Linear:
            return self.shapes[layer_index + 1]
        if layer.type is LayerType.Conv2D:
            shape = self.shapes[layer_index+ 1]
            return (shape[0] * shape[1] * shape[2])        

    # Tracks the shape after the transform corresponding to the layer
    # is applied.
    def update_shape(self, layer):
        if layer.type is LayerType.Linear:
            if len(self.shapes) == 1:
                in_shape = self.shapes.pop()
                self.shapes.append(in_shape[0] * in_shape[1] * in_shape[2])
                self.shapes.append(layer.weight.shape[0])
        elif layer.type is LayerType.Conv2D:
            weight = layer.weight
            num_kernel = weight.shape[0]

            k_h, k_w = layer.kernel_size
            s_h, s_w = layer.stride
            p_h, p_w = layer.padding

            shape = self.shapes[-1]

            input_h, input_w = shape[1:]

            ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
            output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
            output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)
            self.shapes.append((num_kernel, output_h, output_w))
        elif layer.type is LayerType.ReLU:
            if len(self.shapes) == 0:
                raise ValueError("Relu layer should not come at first")
            self.shapes.append(self.shapes[-1])
    

    def back_propagation(self):
        layers_length = len(self.layers)
        diff_struct = None
        lb = None
        ub = None
        for i in reversed(range(layers_length)):
            layer = self.layers[i]
            # concretize the bounds.
            linear_types = [LayerType.Linear, LayerType.Conv2D]
            if diff_struct is not None and layer.type in linear_types:
                curr_lb, curr_ub = self.concrete_substitution()
            if diff_struct is None:
                layer_size = self.get_layer_size(layer_index=i)
                lb_coef = torch.eye(n=layer_size, device=self.device)
                lb_bias = torch.zeros(layer_size, device=self.device)
                ub_coef = torch.eye(n=layer_size, device=self.device)
                ub_bias = torch.zeros(layer_size, device=self.device)
                diff_struct = DeepPolyStruct(lb_bias=lb_bias, lb_coef=lb_coef,
                                             ub_bias=ub_bias, ub_coef=ub_coef)

    def handle_linear(self, layer, last_layer=False):
        # consider two cases.
        if last_layer is True:
            pass
        self.layers.append(layer)


    def handle_conv2d(self, layer):
        pass

    def handle_relu(self, layer):
        self.layers.append(layer)
        return
import torch
import torch.nn.functional as F

class DiffDeepPoly:
    def __init__(self, input1, input2, net, lb_input1, ub_input1, lb_input2, ub_input2) -> None:
        self.input1 = input1
        self.input2 = input2
        self.net = net
        self.lb_input1 = lb_input1
        self.ub_input1 = ub_input1
        self.lb_input2 = lb_input2
        self.ub_input2 = ub_input2
        self.diff = input1 - input2

    # Bias cancels out (Ax + b - Ay - b) = A(x - y) = A * \delta 
    def handle_linear(self, linear_wt, bias, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias):
        delta_lb_bias = delta_lb_bias + (0 if bias is None else delta_lb_coef.matmul(bias))
        delta_ub_bias = delta_ub_bias + (0 if bias is None else delta_ub_coef.matmul(bias))
        
        delta_lb_coef = delta_lb_coef.matmul(linear_wt)
        delta_ub_coef = delta_ub_coef.matmul(linear_wt)

        return delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias

    # preconv shape is the shape before the convolution is applied.
    # postconv shape is the shape after the convolution is applied.
    # while back prop the delta coef shape [rows, postconv shape after flattening].
    def handle_conv(self, conv_weight, conv_bias, delta_lb_coef, delta_lb_bias, 
                    delta_ub_coef, delta_ub_bias, preconv_shape, postconv_shape,
                    stride, padding, groups, dilation):
        kernel_hw = conv_weight.shape[-2:]
        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = delta_lb_coef.shape
        delta_lb_coef = delta_lb_coef.view((coef_shape[0], *postconv_shape))
        delta_ub_coef = delta_ub_coef.view((coef_shape[0], *postconv_shape))

        delta_lb_bias = delta_lb_bias + (0 if conv_bias is None else (delta_lb_coef.sum((3, 4)) * conv_bias).sum(2))
        delta_ub_bias = delta_ub_bias + (0 if conv_bias is None else (delta_ub_coef.sum((3, 4)) * conv_bias).sum(2))

        new_delta_lb_coef = F.conv_transpose2d(delta_lb_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_ub_coef = F.conv_transpose2d(delta_ub_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        
        new_delta_lb_coef = new_delta_lb_coef.view((coef_shape[0], -1))
        new_delta_ub_coef = new_delta_ub_coef.view((coef_shape[0], -1))

        return new_delta_lb_coef, delta_lb_bias, new_delta_ub_coef, delta_ub_bias



    # Consider cases based on the state of the relu for different propagation.
    def handle_relu(self, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias, 
                    lb_input1_layer, ub_input1_layer, lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):
        pass


    def back_substitution(self):
        pass


    def run(self):
        pass
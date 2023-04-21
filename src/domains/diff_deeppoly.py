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

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef))
        return neg_comp, pos_comp
    

    def concretize_bounds(self, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias,
                           delta_lb_layer, delta_ub_layer):
        pass



    # Consider cases based on the state of the relu for different propagation.
    def handle_relu(self, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias, 
                    lb_input1_layer, ub_input1_layer, lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):
        
        input1_active = (lb_input1_layer >= 0)
        input1_passive = (ub_input1_layer <= 0)
        input1_unsettled = (lb_input1_layer < 0 & ub_input1_layer > 0)

        input2_active = (lb_input2_layer >= 0)
        input2_passive = (ub_input2_layer <= 0)
        input2_unsettled = (lb_input2_layer < 0 & ub_input2_layer > 0)

        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = (delta_lb_layer < 0 & delta_ub_layer > 0)


        lambda_lb = torch.zeros(lb_input1_layer.size())
        lambda_ub = torch.zeros(lb_input1_layer.size())
        mu_lb = torch.zeros(lb_input1_layer.size())
        mu_ub = torch.zeros(lb_input1_layer.size())
        # case 1 x.ub <= 0 and y.ub <= 0
        # case 2 x.lb >=  0 and y.lb >= 0
        lambda_lb = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size()), lambda_ub)
        # case 3 x.lb >= 0 and y.ub <= 0
        lambda_lb = torch.where(input1_active & input2_passive, torch.zeros(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_passive, torch.ones(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_active & input2_passive, lb_input1_layer, mu_lb)
        mu_ub = torch.where(input1_active & input2_passive, ub_input2_layer, mu_ub)

        #case 4 (x.lb < 0 and x.ub > 0) and y.ub <= 0
        lambda_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_passive, torch.ones(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size()), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_passive, ub_input2_layer, mu_ub)

        #case 5 (x.ub <= 0) and y.lb >= 0
        lambda_lb = torch.where(input1_passive & input2_active, torch.ones(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_passive & input2_active, torch.zeros(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_passive & input2_active, -ub_input1_layer, mu_lb)
        mu_ub = torch.where(input1_passive & input2_active, -lb_input2_layer, mu_ub)

        # case 6 (x.ub <= 0) and (y.lb < 0 and y.ub > 0)
        lambda_lb = torch.where(input1_passive & input2_unsettled, torch.ones(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_passive & input2_unsettled, torch.zeros(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_passive & input2_unsettled, -ub_input1_layer, mu_lb)
        mu_ub = torch.where(input1_passive & input2_unsettled, torch.zeros(lb_input1_layer.size()), mu_ub)     

        # case 7 (x.lb >= 0) and (y.lb < 0 and y.ub > 0)
        lambda_lb = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_active & input2_unsettled, torch.min(lb_input1_layer, delta_lb_layer), mu_lb)
        mu_ub = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size()), mu_ub)

        # case 8 (x.lb < 0 and x.ub > 0) and (y.lb >= 0)
        lambda_lb = torch.where(input2_unsettled & input2_active, torch.ones(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input2_unsettled & input2_active, torch.zeros(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input2_unsettled & input2_active, torch.zeros(lb_input1_layer.size()), mu_lb)
        mu_ub = torch.where(input2_unsettled & input2_active, torch.max(delta_ub_layer, -lb_input2_layer), mu_ub)

        # case 9 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb >= 0)
        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.ones(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size()), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size()), mu_ub)

        # case 10 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_ub <= 0)
        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.ones(lb_input1_layer.size()), lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size()), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size()), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size()), mu_ub)
        
        # case 11 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb < 0 and delta_ub > 0)
        temp_mu = (delta_lb_layer * delta_ub_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_lb = (-delta_lb_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        temp_lambda_ub = delta_ub_layer / (delta_ub_layer - delta_lb_layer + 1e-15)
        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, -temp_mu, mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_mu, mu_ub)

        # Segregate the +ve and -ve components of the coefficients
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(delta_ub_coef)

        delta_lb_coef = pos_comp_lb * lambda_lb + neg_comp_lb * lambda_ub
        delta_lb_bias = pos_comp_lb * mu_lb + neg_comp_lb * mu_ub
        delta_ub_coef = pos_comp_ub * lambda_ub + neg_comp_ub * lambda_lb
        delta_ub_bias = pos_comp_ub * mu_ub + neg_comp_ub * mu_lb

        return delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias



    def back_substitution(self):
        pass


    def run(self):
        pass
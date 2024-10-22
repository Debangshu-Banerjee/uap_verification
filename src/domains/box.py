import torch
from torch.nn import functional as F
from src.sparsification_util import get_sparsification_indices, prune_last_layer
from copy import deepcopy

class BoxTransformer:
    def __init__(self, prop, complete=False):
        """
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        """      

        self.lbs = [prop.input_lb]
        self.ubs = [prop.input_ub]
        
        self.prop = prop

        self.size = prop.get_input_size()
        self.unstable_relus = []
        self.active_relus = []
        self.inactive_relus = []
        self.perturbation_scaling = {}

        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)

    def compute_lb(self, adv_label=None, complete=False):
        """
        return the lower bound for the variables of the current layer
        """
        if adv_label is not None:
            raise ValueError("Expected adv label to be none")
        if complete:
            lb = self.lbs[-1]
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
            return lb, True, None
        return self.lbs[-1]
        
    def compute_ub(self):
        """
        return the upper bound for the variables of the current layer
        """
        return self.ubs[-1]

    def get_active_relu_list(self):
        return self.active_relus

    def get_inactive_relu_list(self):
        return self.inactive_relus

    # Find the scaling factor to scale perturbation bound.
    def get_perturbation_scaling(self, layer_index):
        if layer_index not in [-1, -2]:
            raise ValueError("Perturbation scaling is not implemented for any layer other than last two layers")
        if layer_index not in self.perturbation_scaling.keys():
            return None
        else:
            return self.perturbation_scaling[layer_index]

    # Populate the scaling factor for perturbation for different
    # index.
    def populate_perturbation_scaling_factor(self, last_layer_wt, output_specification_mat):
        if output_specification_mat is None:
            self.perturbation_scaling[-1] = None
        else:
            # self.perturbation_scaling[-1] = torch.max(torch.norm(output_specification_mat, dim=0))
            self.perturbation_scaling[-1] = 1.0
        if last_layer_wt is None:
            self.perturbation_scaling[-2] = None
        else:
            self.perturbation_scaling[-2] = torch.max(torch.norm(last_layer_wt, dim=0))


    def bound(self):
        return self.compute_lb(), self.compute_ub()
    
    def set_bound(self, lb, ub):
        self.lbs.append(lb)
        self.ubs.append(ub)
    
    def get_all_bounds(self):
        return self.lbs, self.ubs

    def get_layer_bound(self, index):
        lbs, ubs = self.get_all_bounds()
        try:
            return lbs[index], ubs[index]
        except:
            raise ValueError("Index out of bound")
    
    def verify_property_with_pruned_layer(self, pruned_final_layer, adv_label, complete):
        weight = pruned_final_layer.weight.T
        bias = pruned_final_layer.bias
        weight = weight @ self.prop.output_constr_mat()
        bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()        
        weight_pos = F.relu(weight)
        weight_neg = -F.relu(-weight)
        old_lb, old_ub = self.get_layer_bound(-2)
        lb = old_lb @ weight_pos + old_ub @ weight_neg + bias
        if complete:
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
        if lb is not None and torch.all(lb >= 0):
            return True
        else:
            return False

    
    def extract_abstract_features(self, zero_feature_indices, nonzero_feture_indices, 
                                    final_layer, adv_label, complete):
        prune_last_layer(final_layer.weight, zero_feature_indices)
        initial_sparsity = nonzero_feture_indices.size()[0]
        pruned_feture_count = 0
        l = 0
        r = initial_sparsity - 1
        while l <= r:
            mid = (l + r) // 2
            if mid <= 0:
                break
            final_layer_copy = deepcopy(final_layer)
            indices_to_prune = nonzero_feture_indices[:mid]
            prune_last_layer(final_layer_copy.weight, indices_to_prune)
            verification_res = self.verify_property_with_pruned_layer(final_layer_copy, adv_label, complete)
            if verification_res:
                pruned_feture_count = max(pruned_feture_count, mid)
                l = mid + 1
            else:
                r = mid - 1
        optimal_sparsity = initial_sparsity - pruned_feture_count
        return optimal_sparsity



    def compute_sparsification(self, final_layer, adv_label, complete):
        f_lbs, f_ubs = self.get_layer_bound(-2)
        nozero_count, zero_feature_indices, nonzero_feature_indices = get_sparsification_indices(f_lbs, 
                                                f_ubs, final_layer.weight, self.prop.output_constr_mat())
        
        optimal_sparsity = self.extract_abstract_features(zero_feature_indices, nonzero_feature_indices, 
                                                final_layer, adv_label, complete)
        sparsification_result = {}
        sparsification_result["Initial sparsity"] = nozero_count
        sparsification_result["Optimal Sparsity"] = optimal_sparsity
        sparsification_result["zero indices"] = zero_feature_indices
        sparsification_result["Indices prune"] = nonzero_feature_indices[:(nozero_count - optimal_sparsity)]
        sparsification_result["Remaining indices"] = nonzero_feature_indices[(nozero_count - optimal_sparsity):]
        return sparsification_result

    def handle_normalization(self, layer):
        """
        only change the lower/upper bound of the input variables
        """
        mean = layer.mean.view((1, 1))
        sigma = layer.sigma.view((1, 1))
        self.prop.input_lb = (self.prop.input_lb - mean) / sigma
        self.prop.input_ub = (self.prop.input_ub - mean) / sigma

    def handle_addition(self, layer, true_label=None):
        """
        handle addition layer
        """
        bias = layer.bias
        if true_label is not None:
            bias = bias[true_label] - bias

        old_lb, old_ub = self.bound()

        lb = old_lb + bias
        ub = old_ub + bias

        self.set_bound(lb, ub)

    def handle_linear(self, layer, last_layer=False):
        """
        handle linear layer
        """
        weight = layer.weight.T
        bias = layer.bias

        if last_layer:
            weight = weight @ self.prop.output_constr_mat()
            bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()
            self.populate_perturbation_scaling_factor(weight, self.prop.output_constr_mat())

        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        old_lb, old_ub = self.bound()

        weight_pos = F.relu(weight)
        weight_neg = -F.relu(-weight)

        lb = old_lb @ weight_pos + old_ub @ weight_neg + bias
        ub = old_ub @ weight_pos + old_lb @ weight_neg + bias
        if len(lb.shape) == 3:
            lb = torch.diagonal(lb, 0, dim1=0, dim2=1).permute(1, 0)
            ub = torch.diagonal(ub, 0, dim1=0, dim2=1).permute(1, 0)

        self.set_bound(lb, ub)

    def handle_conv2d(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]

        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        old_lb, old_ub = self.bound()

        old_lb = old_lb.reshape(1, shape[0], input_h, input_w)
        old_ub = old_ub.reshape(-1, shape[0], input_h, input_w)

        pos_wt = F.relu(weight)
        neg_wt = -F.relu(-weight)

        lb1 = F.conv2d(old_lb, pos_wt, padding=layer.padding, stride=layer.stride) + F.conv2d(old_ub, neg_wt,
                                                                                              padding=layer.padding,
                                                                                              stride=layer.stride,
                                                                                              bias=bias)

        ub1 = F.conv2d(old_ub, pos_wt, padding=layer.padding, stride=layer.stride) + F.conv2d(old_lb, neg_wt,
                                                                                              padding=layer.padding,
                                                                                              stride=layer.stride,
                                                                                              bias=bias)

        lb = lb1.flatten()
        ub = ub1.flatten()

        self.set_bound(lb, ub)

    def handle_relu(self, layer, optimize=False, relu_mask=None):
        """
        handle relu func
        """
        size = self.size
        lb, ub = self.bound()

        layer_no = len(self.unstable_relus)
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0))[0].tolist())
        self.active_relus.append(torch.where(lb > 0)[0].tolist())
        self.inactive_relus.append(torch.where(ub <= 0)[0].tolist())

        c1_decision = torch.zeros(size, dtype=torch.bool)
        c2_decision = torch.zeros(size, dtype=torch.bool)

        if relu_mask is not None:
            for relu in relu_mask.keys():
                if relu[0] == layer_no:
                    if relu_mask[relu] == 1:
                        c1_decision[relu[1]] = 1
                    elif relu_mask[relu] == -1:
                        c2_decision[relu[1]] = 1

        out_lb = F.relu(lb) * ((~c1_decision) & (~c2_decision))
        out_ub = F.relu(ub) * (~c2_decision)

        self.set_bound(out_lb, out_ub)

    def verify_robustness(self, y, true_label):
        pass


def absmul(lb, ub, weight, bias, down = True):
    """
    Absdomain multiplication
    """
    pos_wgt = F.relu(weight)
    neg_wgt = -F.relu(-weight)
    
    if down:
        new_ilb = lb @ pos_wgt + ub @ neg_wgt
        return new_ilb
    else:
        new_iub = ub @ pos_wgt + lb @ neg_wgt
        return new_iub


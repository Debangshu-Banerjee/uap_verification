import gurobipy as grb
import torch
from torch import nn
from src.common.network import Network, LayerType
import numpy as np

class UAPLPtransformer:
    def __init__(self, mdl, xs, eps, x_lbs, x_ubs, d_lbs, d_ubs, constraint_matrices):
        self.mdl = mdl
        self.xs = xs
        self.batch_size = len(xs)
        # maximum allowed perturbation.
        self.eps = eps
        self.x_lbs = x_lbs
        self.x_ubs = x_ubs
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        # this tracks the index of the last linear (fully connected or conv) 
        # layer already. encountered its -1 at the beginning.
        self.linear_layer_idx = -1 
        self.gmdl = grb.Model()
        self.gurobi_variables = []
        self.debug_log_filename = './debug_logs/uap_lp_debug_log.txt'
        self.debug = True
        self.debug_log_file = None
        self.constraint_matrices = constraint_matrices
        self.tolerence = 1e-4


    def create_lp(self):
        if self.batch_size <= 0:
            return
        self.debug_log_file = open(self.debug_log_filename, 'w+')
        self.gmdl.setParam('OutputFlag', False)
        self.create_constraints()

    def optimize_lp(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        final_vars = []
        final_min_vars = []
        for i in range(self.batch_size):
            constraint_mat = self.constraint_matrices[i]
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
        problem_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                        name='problem_min')
        self.gmdl.addGenConstrMax(problem_min, final_min_vars)
        self.gmdl.setObjective(problem_min, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        if self.gmdl.status == 2:
            self.debug_log_file.write(f"Problem min {problem_min.X}\n")
            # print(f"Problem min {problem_min.X}\n")
            self.debug_log_file.close()
            return problem_min.X
        else:
            print("Gurobi model status", self.gmdl.status)
            self.gmdl.computeIIS()
            self.gmdl.write("model.ilp") 
            self.debug_log_file.close()           
            return NotImplementedError
        
    
    def optimize_milp_percent(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        bs = []
        final_vars = []
        final_min_vars = []
        for i, constraint_mat in enumerate(self.constraint_matrices):
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))
            # self.gmdl.addConstr(bs[idx] == (t_min >= 0))
            BIG_M = 1e8

            # Force bs[-1] to be '1' when t_min > 0
            self.gmdl.addConstr(BIG_M * bs[-1] >= final_var_min)

            # Force bs[-1] to be '0' when t_min < 0 or -t_min  > 0
            self.gmdl.addConstr(BIG_M * (bs[-1] - 1) <= final_var_min)
        
        p = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(p == grb.quicksum(bs[i] for i in range(self.batch_size)) / self.batch_size)
        # self.gmdl.reset()
        self.gmdl.update()
        self.gmdl.setObjective(p, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        if self.gmdl.status == 2:
            self.debug_log_file.write(f"proportion {p.X}\n")
            # print(f"verified proportion {p.X}\n")
            self.debug_log_file.close()
            return p.X
        else:
            print("Gurobi model status", self.gmdl.status)
            print("The optimization failed\n")
            print("Computing computeIIS")
            self.gmdl.computeIIS()
            print("Computing computeIIS finished")            
            self.gmdl.write("model.ilp")
            self.debug_log_file.close()
            return NotImplementedError

    def create_input_constraints(self):
        # the uap perturbation.
        if len(self.xs) <= 0:
            return
        delta = self.gmdl.addMVar(self.xs[0].shape[0], lb = -self.eps, ub = self.eps, vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        vs = [self.gmdl.addMVar(self.xs[i].shape[0], lb = self.xs[i].numpy() - self.eps, ub = self.xs[i].numpy() + self.eps, vtype=grb.GRB.CONTINUOUS, name=f'input_{i}') for i in range(self.batch_size)]
        # ensure all inputs are perturbed by the same uap delta.
        for i, v in enumerate(vs):
            self.gmdl.addConstr(v == self.xs[i].numpy() + delta)
        # ds = [[self.gmdl.addMVar(self.xs[i].shape[0], lb= self.xs[i].numpy() - self.xs[j].numpy()-self.tolerence, ub=self.xs[i].numpy() - self.xs[j].numpy()+self.tolerence, vtype=grb.GRB.CONTINUOUS, name=f'input({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        # for i in range(self.batch_size):
        #     for j in range(i+1, self.batch_size):
        #         self.gmdl.addConstr(ds[i][j - i -1] == self.xs[i].numpy() - self.xs[j].numpy())                
        
        self.gurobi_variables.append({'delta': delta, 'vs': vs, 'ds': None})

    def create_constraints(self):
        self.create_input_constraints()
        layers = self.mdl
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                self.linear_layer_idx += 1
                self.create_linear_constraints(layer, layer_idx)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(layer_idx)
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1                
                self.create_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")

    def create_vars(self, layer_idx, layer_type=''):
        if layer_type == 'linear':            
            vs = [self.gmdl.addMVar(self.x_lbs[i][self.linear_layer_idx].shape[0], lb = self.x_lbs[i][self.linear_layer_idx] - 10.0, ub = self.x_ubs[i][self.linear_layer_idx] + 10.0, vtype=grb.GRB.CONTINUOUS, name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][self.linear_layer_idx].shape[0], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        elif layer_type == 'relu':
            vs = [self.gmdl.addMVar(self.x_lbs[i][self.linear_layer_idx].shape[0], lb =0.0, ub = np.maximum(self.x_ubs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])), vtype=grb.GRB.CONTINUOUS, name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][self.linear_layer_idx].shape[0], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        else:
            raise ValueError(f'layer type {layer_type} is supported yet')

        return vs, ds
    
    def create_linear_constraints(self, layer, layer_idx):
        weight, bias = layer.weight, layer.bias
       
        weight = weight.numpy()
        bias = bias.numpy()
        vs, ds = self.create_vars(layer_idx, 'linear')
        
        for v_idx, v in enumerate(vs):
            self.gmdl.addConstr(v == weight @ self.gurobi_variables[-1]['vs'][v_idx] + bias)

        # if self.debug:
        #     self.gmdl.setObjective(vs[0][1], grb.GRB.MINIMIZE)
        #     self.gmdl.optimize()
        #     v_lb = vs[0][1].X
        #     base_lb = self.x_lbs[0][self.linear_layer_idx][1]
        #     base_ub = self.x_ubs[0][self.linear_layer_idx][1]            
        #     self.gmdl.setObjective(vs[0][1], grb.GRB.MAXIMIZE)
        #     self.gmdl.optimize()
        #     v_ub = vs[0][1].X
        #     self.debug_log_file.write(f"lb : {v_lb} ub : {v_ub}\n\n")
        #     self.debug_log_file.write(f"baseline lb  : {base_lb} ub : {base_ub}\n\n")

        
        for i in range(self.batch_size):
            for j in range(i+1, self.batch_size):
                self.gmdl.addConstr(vs[i] - vs[j] <= (self.d_ubs[(i, j)][self.linear_layer_idx].numpy() + self.tolerence))
                self.gmdl.addConstr(vs[i] - vs[j] >= (self.d_lbs[(i, j)][self.linear_layer_idx].numpy() - self.tolerence))

        if self.linear_layer_idx > 0:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] <= weight @ self.gurobi_variables[-1]['ds'][i][j - i -1] + self.tolerence)
                    self.gmdl.addConstr(ds[i][j - i - 1] >= weight @ self.gurobi_variables[-1]['ds'][i][j - i -1] - self.tolerence)        
        self.gurobi_variables.append({'vs': vs, 'ds': ds})


    def create_relu_ub(self, x, lb, ub): #probably breaks on degenerate lb=ub case, should fix
        rlb, rub = np.max(0, lb), np.max(0, ub)
        return (rub-rlb)/(ub-lb) * (x-lb) + rlb

    def create_relu_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'relu')

        for i in range(self.batch_size):
            self.gmdl.addConstr(vs[i] >= 0)
            self.gmdl.addConstr(vs[i] >= self.gurobi_variables[-1]['vs'][i])
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0] 
            for j in range(tensor_length):                    
                if self.x_lbs[i][self.linear_layer_idx][j] >= 0:
                    self.gmdl.addConstr(vs[i][j] <= self.gurobi_variables[-1]['vs'][i][j])
                    continue 
                elif self.x_ubs[i][self.linear_layer_idx][j] <= 0:
                    self.gmdl.addConstr(vs[i][j] <= 0)
                    continue
                else:
                    ub = self.x_ubs[i][self.linear_layer_idx][j]
                    lb = self.x_lbs[i][self.linear_layer_idx][j]
                    slope = ub / (ub - lb + 1e-15)
                    mu = -slope * lb
                    self.gmdl.addConstr(vs[i][j] <= slope * self.gurobi_variables[-1]['vs'][i][j] + mu)
        
        for i in range(self.batch_size):
            for j in range(i+1, self.batch_size):
                tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                for k in range(tensor_length):
                    # case 1 x unsettled & y passive
                    x_active = (self.x_lbs[i][self.linear_layer_idx][k] >= 0)
                    x_passive = (self.x_ubs[i][self.linear_layer_idx][k] <= 0)
                    x_unsettled = (~x_active) & (~x_passive)
                    y_active = (self.x_lbs[j][self.linear_layer_idx][k] >= 0)
                    y_passive = (self.x_ubs[j][self.linear_layer_idx][k] <= 0)
                    y_unsettled = (~y_active) & (~y_passive)
                    delta_active = (self.d_lbs[(i, j)][self.linear_layer_idx][k] >= 0)
                    delta_passive = (self.d_ubs[(i, j)][self.linear_layer_idx][k] <= 0)
                    delta_unsettled = (~delta_active) & (~delta_passive)                                                                                
                    if x_unsettled and y_passive and delta_active:
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                    elif x_unsettled and y_active:
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                    elif x_passive and y_unsettled and delta_passive:
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                    elif x_active and y_unsettled:
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                    elif x_unsettled and y_unsettled and delta_active:
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= 0.0)                        
                    elif x_unsettled and y_unsettled and delta_passive:
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= 0.0)
                    elif x_unsettled and y_unsettled and delta_unsettled:
                        temp_lb = self.d_lbs[(i, j)][self.linear_layer_idx][k]
                        temp_ub = self.d_ubs[(i, j)][self.linear_layer_idx][k]
                        d_lambda_lb = temp_lb / (temp_lb - temp_ub + 1e-15)
                        d_lambda_ub = temp_ub / (temp_ub - temp_lb + 1e-15)
                        d_mu_lb = d_lambda_ub * temp_lb
                        d_mu_ub = -d_lambda_ub * temp_lb
                        self.gmdl.addConstr(ds[i][j - i -1][k] >= d_lambda_lb * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_lb)
                        self.gmdl.addConstr(ds[i][j - i -1][k] <= d_lambda_ub * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_ub)                                                
        # We add lp formulation for computing differences in the next iterations.
        # for k in range(len(ds)): #this seems terrible and slow, ill speed it up
        #     for l in range(len(ds[k])):
        #         d = ds[k][l].tolist()
        #         r = self.d_idx[k][l]
        #         for i in range(len(d)):
        #             for j in range(len(d[0])): #check constraints
        #                 if self.x_ubs[layer_idx][k][i][j] <= 0 and self.x_ubs[layer_idx][r][i][j] <= 0:
        #                     self.gmdl.addConstr(d[i][j] == 0)
        #                 #elif self.x1_u[layer_idx][i][j] <= 0 and self.x2_l[layer_idx][i][j] >= 0:
        #                 #    self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['v2'])
        #                 #elif self.x1_l[layer_idx][i][j] >= 0 and self.x2_u[layer_idx][i][j] <= 0:
        #                 #    self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['v1'])
        #                 elif self.x_lbs[layer_idx][k][i][j] >= 0 and self.x_lbs[layer_idx][r][i][j] >= 0:
        #                     self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs']['r'])
        #                 elif self.x_ubs[layer_idx][k][i][j] <= 0:
        #                     self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['vs'][r])
        #                 #elif self.x1_l[layer_idx][i][j] >= 0:
        #                 elif self.x_ubs[layer_idx][r][i][j] <= 0:
        #                     self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['vs'][k])
        #                 #elif self.x2_l[layer_idx][i][j] >= 0:
        #                 else:
        #                     self.gmdl.addConstr(d[i][j] >= 0)
        #                     self.gmdl.addConstr(d[i][j] >= self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs'][r])
        #                     self.gmdl.addConstr(d[i][j] <= self.create_relu_ub(self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs'][r], self.d_lbs[layer_idx-1][i][j], self.d_ubs[layer_idx-1][i][j])) #not sure
                            
        self.gurobi_variables.append({'vs': vs, 'ds': ds})
        
    def create_conv2d_constraints(self, layer, layer_idx):
        raise NotImplementedError

    def get_layer_type(self, layer):
        return layer.type

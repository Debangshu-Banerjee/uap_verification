import gurobipy as grb
import torch
from torch import nn
from src.common.network import Network, LayerType
import numpy as np

class UAPLPtransformer:
    def __init__(self, mdl, x1, x2, x1_l, x1_u, x2_l, x2_u, d_l, d_u):
        self.mdl = mdl
        self.x1 = x1
        self.x2 = x2
        self.x1_l = x1_l
        self.x1_u = x1_u
        self.x2_l = x2_l
        self.x2_u = x2_u
        self.d_l = d_l
        self.d_u = d_u

        self.gmdl = grb.Model()
        self.gurobi_variables = []

    def create_lp(self):
        self.gmdl.setParam('OutputFlag', False)
        self.create_constraints()

    def optimize_lp(self, y1, y2):
        bounds_y1 = []
        for i in range(len(self.gurobi_variables[-1]['v1'])):
            optimize_var = self.gurobi_variables[-1]['v1'][i]
            self.model.reset()
            if i != y1:
                self.model.setObjective(optimize_var, grb.GRB.MAXIMIZE)
            else:
                self.model.setObjective(optimize_var, grb.GRB.MAXIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                bounds_y1.append(optimize_var.X)
            else:
                raise NotImplementedError
        bounds_y2 = []
        for i in range(len(self.gurobi_variables[-1]['v2'])):
            optimize_var = self.gurobi_variables[-1]['v2'][i]
            self.model.reset()
            if i != y2:
                self.model.setObjective(optimize_var, grb.GRB.MAXIMIZE)
            else:
                self.model.setObjective(optimize_var, grb.GRB.MAXIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                bounds_y2.append(optimize_var.X)
            else:
                raise NotImplementedError
        return np.argmax(bounds_y1) == y1 and np.argmax(bounds_y2) == y2

    def create_constraints(self):
        layers = self.mdl
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)

            if layer_type == LayerType.Linear:
                self.create_linear_constraints(layer, layer_idx)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(layer_idx)
            elif layer_type == LayerType.Conv2D:
                self.create_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")

    def create_vars(self, layer_idx):
        v1 = self.gmdl.addMVar(self.x1_l[layer_idx].shape, lb = self.x1_1[layer_idx], ub = self.x1_u[layer_idx], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_x1')
        v2 = self.gmdl.addMVar(self.x2_l[layer_idx].shape, lb = self.x2_1[layer_idx], ub = self.x2_u[layer_idx], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_x2')
        d = self.gmdl.addMVar(self.d_l[layer_idx].shape, lb = self.d_1[layer_idx], ub = self.d_u[layer_idx], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d')
        return v1, v2, d

    def create_linear_constraints(self, layer, layer_idx):
        weight, bias = layer.weight, layer.bias

        v1, v2, d = self.create_vars(layer_idx)

        if layer_idx != 0:
            self.gmdl.addConstr(v1 == weight @ self.gurobi_variables[-1]['v1'] + bias)
            self.gmdl.addConstr(v2 == weight @ self.gurobi_variables[-1]['v2'] + bias)
            self.gmdl.addConstr(d == weight @ (self.gurobi_variables[-1]['v1'] - self.gurobi_variables[-1]['v2']))

        self.gurobi_variables.append({'v1': v1, 'v2': v2, 'd': d})


    def create_relu_constraints(self, layer_idx):
        v1, v2, d = self.create_vars(layer_idx)

        self.gmdl.addConstr(v1 >= 0)
        self.gmdl.addConstr(v1 >= self.gurobi_variables[-1]['v1'])
        self.gmdl.addConstr(v2 >= 0)
        self.gmdl.addConstr(v2 >= self.gurobi_variables[-1]['v2'])
        
        d = d.tolist()
        for i in range(len(d)):
            for j in range(len(d[0])):
                if self.x1_u[layer_idx][i][j] <= 0 and self.x2_u[layer_idx][i][j] <= 0:
                    self.gmdl.addConstr(d[i][j] == 0)
                #elif self.x1_u[layer_idx][i][j] <= 0 and self.x2_l[layer_idx][i][j] >= 0:
                #    self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['v2'])
                #elif self.x1_l[layer_idx][i][j] >= 0 and self.x2_u[layer_idx][i][j] <= 0:
                #    self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['v1'])
                elif self.x1_l[layer_idx][i][j] >= 0 and self.x2_l[layer_idx][i][j] >= 0:
                    self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['v1'] - self.gurobi_variables[-1]['v2'])
                elif self.x1_u[layer_idx][i][j] <= 0:
                    self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['v2'])
                #elif self.x1_l[layer_idx][i][j] >= 0:
                elif self.x2_u[layer_idx][i][j] <= 0:
                    self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['v1'])
                #elif self.x2_l[layer_idx][i][j] >= 0:
                else:
                    self.gmdl.addConstr(d[i][j] >= 0)
                    self.gmdl.addConstr(d[i][j] >= self.gurobi_variables[-1]['v1'] - self.gurobi_variables[-1]['v2'])
                    
        self.gurobi_variables.append({'v1': v1, 'v2': v2, 'd': d})
        
    def create_conv2d_constraints(self, layer, layer_idx):
        raise NotImplementedError

    def get_layer_type(self, layer):
        if self.format == "onnx":
            return layer.type

        if self.format == "torch":
            if type(layer) is nn.Linear:
                return LayerType.Linear
            elif type(layer) is nn.Conv2d:
                return LayerType.Conv2D
            elif type(layer) == nn.ReLU:
                return LayerType.ReLU
            elif type(layer) == nn.Flatten():
                return LayerType.Flatten
            else:
                return LayerType.NoOp
                # raise ValueError("Unsupported layer type for torch model!", type(layer))

        raise ValueError("Unsupported model format or model format not set!")
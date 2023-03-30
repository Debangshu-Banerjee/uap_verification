import torch
import gurobipy as grb
import numpy as np


class ZonoUAPTransformer:
    def __init__(self, prop, complete=False):
        self.prop = prop
        self.complete = complete
        self.coefs = None
        self.centers = None
        self.input_size = self.prop.get_input_size()
    
    def set_coefs_centers(self, coefs, centers):
        self.coefs = coefs
        self.centers = centers

    def compute_lb(self):
        if self.coefs is None or self.centers is None:
            raise ValueError("coefs or center is NULL.")
        assert len(self.coefs) == len(self.centers)
        actual_coefs = []
        lbs = []
        actual_lbs = []
        for i, coef in enumerate(self.coefs):
            center = self.centers[i]
            input_coefs = coef[:self.input_size]
            input_coefs = input_coefs.T
            other_coefs = coef[self.input_size:]
            cof_abs = torch.sum(torch.abs(other_coefs), dim=0)
            lb = center - cof_abs
            lbs.append(lb.detach().numpy())
            actual_coefs.append(input_coefs)
            cof_abs = torch.sum(torch.abs(input_coefs), dim=1)
            actual_lbs.append(-cof_abs)
        with grb.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with grb.Model(env=env) as m:
                epsilons = m.addMVar(self.input_size, lb=-1, ub=1, name='epsilons')
                final_val = m.addVar(lb=float('-inf'), ub=float('inf'), name=f'objective_vars_{i}')
                all_t = []  
                for i, input_coefs in enumerate(actual_coefs):
                    input_coefs = input_coefs.detach().numpy()
                    t = m.addMVar(input_coefs.shape[0], lb=float('-inf'), ub=float('inf'), name=f'individual_lbs_{i}')
                    all_t.append(t)
                    m.addConstr(input_coefs @ epsilons + lbs[i] == t)
                    for i in range(input_coefs.shape[0]):
                        m.addConstr(t[i] <= final_val)
                    m.update()
                    m.setObjective(final_val, grb.GRB.MINIMIZE)
                    _ = m.optimize()
                    global_lb = final_val.X
                return global_lb  


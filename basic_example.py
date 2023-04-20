import torch
import gurobipy as grb
import numpy as np


def get_l_u(m, input1_l1_in):
    m.setObjective(input1_l1_in[0], grb.GRB.MINIMIZE)
    m.optimize()
    l11 = input1_l1_in[0].X
    m.setObjective(input1_l1_in[1], grb.GRB.MINIMIZE)
    m.optimize()
    l12 = input1_l1_in[1].X
    m.setObjective(input1_l1_in[0], grb.GRB.MAXIMIZE)
    m.optimize()
    u11 = input1_l1_in[0].X
    m.setObjective(input1_l1_in[1], grb.GRB.MAXIMIZE)
    m.optimize()
    u12 = input1_l1_in[1].X
    return l11, l12, u11, u12

def get_constraints(m, input, output, bounds):
    n = len(bounds)
    for i in range(n):
        l = bounds[i][0]
        u = bounds[i][1]
        if l >= 0:
            m.addConstr(input[i] == output[i])
        elif u <= 0:
            m.addConstr(0.0 == output[i])
        else:
            m.addConstr(u*(input[i] - l) / (u - l) >= output[i])
            m.addConstr(-input[i] >= -output[i])
            m.addConstr(0.0 >= -output[i])
    m.update()

def add_differential_constraints(m, layer, center1, center2, input1, input2):
    difference = layer @ (center1 - center2)
    for i, diff in enumerate(difference):
        print("Diff", diff)
        if diff >= 0:
            m.addConstr(input1[i] - input2[i] <= diff)
            m.addConstr(input1[i] - input2[i] >= 0.0)
        else:
            m.addConstr(input1[i] - input2[i] <= 0.0)
            m.addConstr(input1[i] - input2[i] >= diff)
    m.update()


with grb.Env(empty=True) as env:
    env.setParam("OutputFlag", 0)
    env.start()
    with grb.Model(env=env) as m:
        m = grb.Model()
        epsilons = m.addMVar(2, lb=-6, ub=6, name='epsilons')
        layer1 = torch.tensor([[1, -1], [-2, 1]])
        layer2 = torch.tensor([[1,  0.3], [0.5, 1]])
        layer1 = layer1.numpy()
        layer2 = layer2.numpy()
        center1 = torch.tensor([14, 11])
        center1 = center1.numpy()
        input1_l1_in = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='input1_l1_in')
        m.addConstr(layer1 @ epsilons + layer1 @ center1 == input1_l1_in)
        m.update()
        input1_l1_out = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='input1_l1_out')
        l11, l12, u11, u12 = get_l_u(m=m ,input1_l1_in=input1_l1_in)
        print('***' + f'{l11} {l12} {u11} {u12}')
        bounds1 = [[l11, u11], [l12, u12]]
        get_constraints(m=m, input=input1_l1_in, output=input1_l1_out, bounds=bounds1)
        output1 = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='output1')
        m.addConstr(layer2 @ input1_l1_out == output1)
        obj1 = m.addVar(lb=float('-inf'), ub=float('inf'), name=f'obj1')
        m.addConstr(output1[0]  - output1[1] == obj1)
        m.update()

        center2 = torch.tensor([12, 14])
        center2 = center2.numpy()
        input2_l1_in = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='input2_l1_in')
        m.addConstr(layer1 @ epsilons + layer1 @ center2 == input2_l1_in)
        m.update()
        input2_l1_out = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='input2_l1_out')
        l21, l22, u21, u22 = get_l_u(m=m ,input1_l1_in=input2_l1_in)
        bounds2 = [[l21, u21], [l22, u22]]
        get_constraints(m=m, input=input2_l1_in, output=input2_l1_out, bounds=bounds2)
        print('***' + f'{l21} {l22} {u21} {u22}')
        output2 = m.addMVar(2, lb=float('-inf'), ub=float('inf'), name='output1')
        m.addConstr(layer2 @ input2_l1_out == output2)
        obj2 = m.addVar(lb=float('-inf'), ub=float('inf'), name=f'obj2')
        m.addConstr(output2[1] - output2[0] == obj2)
        obj = m.addVar(lb=float('-inf'), ub=float('inf'), name=f'obj')
        m.addConstr(obj1 <= obj)
        m.addConstr(obj2 <= obj)
        m.setObjective(obj, grb.GRB.MINIMIZE)
        m.update()
        m.optimize()
        print(f"objective {obj.X}")
        print(f"objective1 {obj1.X}")        
        print(f"objective2 {obj2.X}")
        print(f"out_in_1 {input1_l1_in.X}")
        print(f"out_in_2 {input2_l1_in.X}")
        print(f"out1 {input1_l1_out.X}")
        print(f"out2 {input2_l1_out.X}")
        obj_wt_diff = obj.X
        add_differential_constraints(m=m, layer=layer1, center1=center1, 
                                     center2=center2, input1=input1_l1_out, input2=input2_l1_out)
        m.update()
        m.optimize()
        print(f"objective {obj.X}")
        print(f"objective1 {obj1.X}")        
        print(f"objective2 {obj2.X}")
        print(f"out_in_1 {input1_l1_in.X}")
        print(f"out_in_2 {input2_l1_in.X}")        
        print(f"out1 {input1_l1_out.X}")
        print(f"out2 {input2_l1_out.X}")
        obj_with_diff = obj.X
        print("Improvement", (obj_with_diff - obj_wt_diff) / (abs(obj_wt_diff) + 1e-8) * 100)

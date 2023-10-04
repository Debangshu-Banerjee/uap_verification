\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 layer0_d(0-1)[0] + 0 layer0_d(0-1)[1] + 0 layer1_d(0-1)[0]
   + 0 layer1_d(0-1)[1] + 0 layer2_d(0-1)[0] + 0 layer2_d(0-1)[1]
   + 0 final_var_min_0 + 0 final_var_min_1 + problem_min
Subject To
 R0: - uap_delta[0] + input_0[0] = 14
 R1: - uap_delta[1] + input_0[1] = 11
 R2: - uap_delta[0] + input_1[0] = 11
 R3: - uap_delta[1] + input_1[1] = 14
 R4: - input_0[0] + input_0[1] + layer_0_linear_x0[0] = 0
 R5: 2 input_0[0] - input_0[1] + layer_0_linear_x0[1] = 0
 R6: - input_1[0] + input_1[1] + layer_0_linear_x1[0] = 0
 R7: 2 input_1[0] - input_1[1] + layer_0_linear_x1[1] = 0
 R8: layer_1_relu_x0[0] >= 0
 R9: layer_1_relu_x0[1] >= 0
 R10: - layer_0_linear_x0[0] + layer_1_relu_x0[0] >= 0
 R11: - layer_0_linear_x0[1] + layer_1_relu_x0[1] >= 0
 R12: - 0.625 layer_0_linear_x0[0] + layer_1_relu_x0[0] <= 5.625
 R13: - 0.0277777779847383 layer_0_linear_x0[1] + layer_1_relu_x0[1]
   <= 0.9722222089767456
 R14: layer_1_relu_x1[0] >= 0
 R15: layer_1_relu_x1[1] >= 0
 R16: - layer_0_linear_x1[0] + layer_1_relu_x1[0] >= 0
 R17: - layer_0_linear_x1[1] + layer_1_relu_x1[1] >= 0
 R18: - 0.375 layer_0_linear_x1[0] + layer_1_relu_x1[0] <= 5.625
 R19: - 0.2777777910232544 layer_0_linear_x1[1] + layer_1_relu_x1[1]
   <= 7.222222328186035
 R20: - layer_1_relu_x0[0] + layer_1_relu_x0[1] + layer_2_linear_x0[0] = 0
 R21: layer_1_relu_x0[0] - layer_1_relu_x0[1] + layer_2_linear_x0[1] = 0
 R22: - layer_1_relu_x1[0] + layer_1_relu_x1[1] + layer_2_linear_x1[0] = 0
 R23: layer_1_relu_x1[0] - layer_1_relu_x1[1] + layer_2_linear_x1[1] = 0
 R24: - layer_2_linear_x0[0] + layer_2_linear_x0[1] + final_var_0[0] = 0
 R25: layer_2_linear_x1[0] - layer_2_linear_x1[1] + final_var_1[0] = 0
Bounds
 -6 <= uap_delta[0] <= 6
 -6 <= uap_delta[1] <= 6
 8 <= input_0[0] <= 20
 5 <= input_0[1] <= 17
 5 <= input_1[0] <= 17
 8 <= input_1[1] <= 20
 -9 <= layer_0_linear_x0[0] <= 15
 -35 <= layer_0_linear_x0[1] <= 1
 -15 <= layer_0_linear_x1[0] <= 9
 -26 <= layer_0_linear_x1[1] <= 10
 5.800000190734863 <= layer0_d(0-1)[0] <= 6.199999809265137
 -9.199999809265137 <= layer0_d(0-1)[1] <= -8.800000190734863
 layer_1_relu_x0[0] <= 15
 layer_1_relu_x0[1] <= 1
 layer_1_relu_x1[0] <= 9
 layer_1_relu_x1[1] <= 10
 -9 <= layer1_d(0-1)[0] <= 15
 -10 <= layer1_d(0-1)[1] <= 1
 -6.624999523162842 <= layer_2_linear_x0[0] <= 15.97222137451172
 -15.97222137451172 <= layer_2_linear_x0[1] <= 6.624999523162842
 -15.625 <= layer_2_linear_x1[0] <= 16.22222137451172
 -16.22222137451172 <= layer_2_linear_x1[1] <= 15.625
 -0.2000000029802322 <= layer2_d(0-1)[0] <= 15.19999980926514
 -15.19999980926514 <= layer2_d(0-1)[1] <= 0.2000000029802322
 final_var_0[0] free
 final_var_min_0 free
 final_var_1[0] free
 final_var_min_1 free
 problem_min free
General Constraints
 GC0: final_var_min_0 = MIN ( final_var_0[0] )
 GC1: final_var_min_1 = MIN ( final_var_1[0] )
 GC2: problem_min = MAX ( final_var_min_0 , final_var_min_1 )
End

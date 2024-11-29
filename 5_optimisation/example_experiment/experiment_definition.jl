"""
Experiment description:
Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi 
ut aliquip ex ea commodo consequat. 
"""

# Definition of the considered problems
exp_problems = ("ruoff_atp_consumption",)
exp_noise_levels = Dict("ruoff_atp_consumption" => [5, 10, 20, 35])
exp_sparsity = Dict("ruoff_atp_consumption" => [50, 100, 150, 200])

# Definition of the NN architecture grid
exp_hidden_layers = (1, 2, 4)
exp_hidden_neurons = (3, 5)
exp_act_fct = ("tanh",)
exp_input_normalization = (true, false)

# Definition of the startpoints
exp_startpoints = Dict("lhs" => 1:200)

# Definition of the optimization pipeline 
exp_reg = ("l2",)
exp_λ_reg = (0.0, 0.01, 0.1, 1.0, 10.0)
exp_lr_adam = (0.01,)
epochs = (500, 3000) # (epochs_adam, epochs_bfgs)

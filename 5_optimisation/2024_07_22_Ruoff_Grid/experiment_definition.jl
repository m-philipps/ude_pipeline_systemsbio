"""
Experiment description:
Ruoff ATP consumption model on the grid of hyperparameters as defined in 3_start_points/ruoff_atp_consumption
For each problem (defined by noise and sparsity level), we optimize the UDE with the same hyperparameters.
"""

hps_file_path = "3_start_points/ruoff_atp_consumption/hps.csv"

# Definition of the considered problems
exp_problems = ("ruoff_atp_consumption",)
exp_noise_levels = Dict("ruoff_atp_consumption" => [5, 10, 20, 35])
exp_sparsity = Dict("ruoff_atp_consumption" => [50, 100, 150, 200])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"

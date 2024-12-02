"""
Experiment description:
Ruoff ATP consumption model on the grid of hyperparameters as defined in 3_start_points/ruoff_atp_consumption
Based on the experiment 2024_07_22_Ruoff_Trid we expand the noise sparsity grid - this experiment runs sparsity 25.
"""

hps_file_path = "3_start_points/ruoff_atp_consumption/hps.csv"

# Definition of the considered problems
exp_problems = ("ruoff_atp_consumption", )
exp_noise_levels = Dict("ruoff_atp_consumption" => [5, 10, 20, 35])
exp_sparsity = Dict("ruoff_atp_consumption" => [25])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"
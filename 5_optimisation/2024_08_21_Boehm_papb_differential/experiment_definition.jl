"""
Experiment description:
Run a small grid on the boehm_papb_differential problem.
"""

hps_file_path = "3_start_points/boehm_papb_differential/hps.csv" # use same HP

# Definition of the considered problems
exp_problems = ("boehm_papb_differential", )
exp_noise_levels = Dict("boehm_papb_differential" => ["-"])
exp_sparsity = Dict("boehm_papb_differential" => [16])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"

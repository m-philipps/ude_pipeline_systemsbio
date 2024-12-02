"""
Experiment description:
Run 3000 UDEs on the boehm_papa_export_kinetic problem.
"""

hps_file_path = "3_start_points/boehm_papa_export_kinetic/hps.csv"

# Definition of the considered problems
exp_problems = ("boehm_papa_export_kinetic", )
exp_noise_levels = Dict("boehm_papa_export_kinetic" => ["-"])
exp_sparsity = Dict("boehm_papa_export_kinetic" => [16])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"

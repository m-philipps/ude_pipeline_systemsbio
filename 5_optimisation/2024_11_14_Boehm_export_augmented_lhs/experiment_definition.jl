"""
Experiment description:
Run 3000 UDEs on the boehm_papb_differential_BaF3_Epo problem using the standard LHS for startpoint selection.
"""

hps_file_path = "3_start_points/boehm_export_augmented/hps.csv"

# Definition of the considered problems
exp_problems = ("boehm_export_augmented", )
exp_noise_levels = Dict("boehm_export_augmented" => ["-"])
exp_sparsity = Dict("boehm_export_augmented" => [16])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"

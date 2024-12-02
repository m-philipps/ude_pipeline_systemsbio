"""
Experiment description:
Run 15 000 starts on the observable_ab_ratio problem, using hyperparaemters with comparatively large regularization parameters. 
"""

hps_file_path = "3_start_points/boehm_observable_ab_ratio/hps_largereg.csv"

# Definition of the considered problems
exp_problems = ("boehm_observable_ab_ratio", )
exp_noise_levels = Dict("boehm_observable_ab_ratio" => ["-"])
exp_sparsity = Dict("boehm_observable_ab_ratio" => [16])

# Fixed hyperparameters
epochs = (500, 3000) # (max_epochs_adam, max_epochs_bfgs)
reg_method = "l2"
startpoint_method = "lhs"

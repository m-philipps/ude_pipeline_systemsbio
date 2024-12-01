include("utils.jl")
include("../2_measurements/load_data.jl")
using DataFrames: nrow

experiment_name = "2024_07_22_Ruoff_Grid"
test_setup = false
hps_from_file = true # use the hyperparameters from the file defined in the experiment definition (true) or a grid set-up defined in the respective folder (false)

if test_setup
    slurm_array_nr = 1
else
    slurm_array_nr = parse(Int, ARGS[1]) # each SLURM array corresponds to one UDE
end

# access experimental setting
experimental_settings = create_overview_dataframe(experiment_name, hps_from_file)

if slurm_array_nr == 1
    # store experimental setting once
    store_overview_dataframe(experiment_name, experimental_settings)
end

include(joinpath(pwd(), "4_ude/$(experimental_settings[1,"problem_name"]).jl"))
if length(unique(experimental_settings.problem_name)) > 1
    println("Different problems in one experiment are not supported.")
    exit()
end

# select hyperparameters of current subexperiment
experimental_setting =
    experimental_settings[experimental_settings.ude_nr.==slurm_array_nr, :][1, :]
# optimize
result_path = "."
result_path = joinpath(
    "5_optimisation",
    experiment_name,
    "result",
    "ude_$(experimental_setting["ude_nr"])",
)
train_ude(experimental_setting, result_path)

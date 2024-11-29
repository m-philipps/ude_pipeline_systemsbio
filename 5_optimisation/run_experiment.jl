include("utils.jl")
include("../2_measurements/load_data.jl")
using DataFrames: nrow

experiment_name = ARGS[2]
test_setup = false
local_result_storage = true
marvin = true # marvin cluster settings: one SLURM array represents all experiment arrays of one startpoint
hps_from_file = true # use the hyperparameters from the file defined in the experiment definition (true) or a grid set-up defined in the respective folder (false)

if test_setup
    slurm_array_nr = 1
else
    slurm_array_nr = parse(Int, ARGS[1])
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

if marvin
    n_exp_array_per_slurm_array = Int(nrow(experimental_settings) / 1000)
    slurm_array_start_row = (slurm_array_nr - 1) * n_exp_array_per_slurm_array + 1
    slurm_array_end_row = slurm_array_nr * n_exp_array_per_slurm_array

    for i = slurm_array_start_row:slurm_array_end_row
        # select hyperparameters of current subexperiment
        experimental_setting = experimental_settings[i, :]

        # optimize
        result_path = "."
        if local_result_storage
            result_path = joinpath(
                "5_optimisation",
                experiment_name,
                "result",
                "ude_$(experimental_setting["ude_nr"])",
            )
        else
            result_path = joinpath(
                "/storage/groups/hasenauer_lab/sym",
                experiment_name,
                "result",
                "ude_$(experimental_setting["ude_nr"])",
            )
        end
        try
            println("$(experimental_setting["ude_nr"])")
            train_ude(experimental_setting, result_path)
        catch
            println("Error for UDE $(experimental_setting["ude_nr"])")
        end

    end
else
    # select hyperparameters of current subexperiment
    experimental_setting =
        experimental_settings[experimental_settings.ude_nr.==slurm_array_nr, :][1, :]
    # optimize
    result_path = "."
    if local_result_storage
        result_path = joinpath(
            "5_optimisation",
            experiment_name,
            "result",
            "ude_$(experimental_setting["ude_nr"])",
        )
    else
        result_path = joinpath(
            "/storage/groups/hasenauer_lab/sym",
            experiment_name,
            "result",
            "ude_$(experimental_setting["ude_nr"])",
        )
    end
    train_ude(experimental_setting, result_path)
end

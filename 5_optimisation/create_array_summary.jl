include("utils.jl")
include("../2_measurements/load_data.jl")
using DataFrames: outerjoin
using CSV: write

experiment_name = ARGS[2] # "2024_07_22_Ruoff_Grid"
local_result_storage = true
split_summmary = true

# load overview of experiments
experimental_settings = load_overview_dataframe(experiment_name)

if split_summmary
    array_nr = parse(Int, ARGS[1])
    eval_n_udes = parse(Int, ARGS[3])
    ude_start = 1 + eval_n_udes * (array_nr - 1)
    ude_end = eval_n_udes * array_nr
    experimental_settings = experimental_settings[ude_start:ude_end, :]
    println("Evaluate from $(ude_start) to $(ude_end)")
end

result_path = "."
if local_result_storage
    result_path = joinpath("5_optimisation", experiment_name, "result")
else
    result_path = joinpath("/storage/groups/hasenauer_lab/sym", experiment_name, "result")
end

function get_epochs_and_runtime(ude_nr, result_path)
    training_times_and_epochs = missing
    try
        training_times_and_epochs =
            load(joinpath(result_path, "ude_$(ude_nr)", "training_times_and_epochs.jld2"))
    catch
        return Dict(
            "runtime_ADAM_in_min" => NaN,
            "runtime_BFGS_in_min" => NaN,
            "epochs_ADAM" => NaN,
            "epochs_BFGS" => NaN,
            "epoch_optimum" => NaN,
        )
    end
    runtime_ADAM_in_min = training_times_and_epochs["Adam_time_in_s"] / 60
    runtime_BFGS_in_min = training_times_and_epochs["BFGS_time_in_s"] / 60
    epochs_ADAM = training_times_and_epochs["Adam_iterations"]
    epochs_BFGS = training_times_and_epochs["BFGS_iterations"]

    res = load(joinpath(result_path, "ude_$(ude_nr)", "loss_curves.jld2"))
    epoch_of_optima = argmin(res["validation_losses"])

    return Dict(
        "runtime_ADAM_in_min" => runtime_ADAM_in_min,
        "runtime_BFGS_in_min" => runtime_BFGS_in_min,
        "epochs_ADAM" => epochs_ADAM,
        "epochs_BFGS" => epochs_BFGS,
        "epoch_optimum" => epoch_of_optima,
    )
end

function predict_train_validation_test(experimental_setting, result_path)
    ude_nr = experimental_setting["ude_nr"]
    # set up UDE
    nn_model, ude_dynamics!, observable_mapping = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)

    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)

    # load data
    t_train, y_train, _ = load_measurements(
        experimental_setting["problem_name"],
        experimental_setting["sparsity"],
        experimental_setting["noise_level"],
        "training",
    )
    t_val, y_val, _ = load_measurements(
        experimental_setting["problem_name"],
        experimental_setting["sparsity"],
        experimental_setting["noise_level"],
        "validation",
    )
    test_data = load_reference(experimental_setting["problem_name"])

    if occursin("ruoff", experimental_setting["problem_name"])
        solver = Tsit5()
    else
        solver = KenCarp4()
    end

    # predict 
    IC, IC_ids = get_IC_or_IC_IDS(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting["sparsity"],
        p_opt,
    )
    pred_train, pred_val, pred_test = missing, missing, missing
    if ismissing(IC)
        pred_train =
            predict_hidden_states(t_train, p_opt, st, IC_ids, true, dynamics!, solver)
        pred_val = predict_hidden_states(
            t_val,
            p_opt,
            st,
            IC_ids,
            t_val[1] == t_train[1],
            dynamics!,
            solver,
        )
        if !occursin("boehm", experimental_setting["problem_name"])
            pred_test = predict_hidden_states(
                test_data.time,
                p_opt,
                st,
                IC_ids,
                test_data.time[1] == t_train[1],
                dynamics!,
                solver,
            )
        end
    else
        pred_train =
            predict_hidden_states_given_IC(t_train, p_opt, st, IC, true, dynamics!, solver)
        pred_val = predict_hidden_states_given_IC(
            t_val,
            p_opt,
            st,
            IC,
            t_val[1] == t_train[1],
            dynamics!,
            solver,
        )
        if !occursin("boehm", experimental_setting["problem_name"])
            pred_test = predict_hidden_states_given_IC(
                test_data.time,
                p_opt,
                st,
                IC,
                test_data.time[1] == t_train[1],
                dynamics!,
                solver,
            )
        end
    end

    pred_obs_train = observable_mapping(pred_train, p_opt, st)
    pred_obs_val = observable_mapping(pred_val, p_opt, st)
    if !occursin("boehm", experimental_setting["problem_name"])
        pred_obs_test = observable_mapping(pred_test, p_opt, st)
    end

    if occursin("ruoff", experimental_setting["problem_name"])
        OBSERVABLES_IDS = ["N_2", "A_3"]
    end

    train_res = Dict(
        "t_train" => t_train,
        "prediction_hidden_train" => pred_train,
        "prediction_obs_train" => pred_obs_train,
        "obs_train" => y_train,
    )
    val_res = Dict(
        "t_val" => t_val,
        "prediction_hidden_val" => pred_val,
        "prediction_obs_val" => pred_obs_val,
        "obs_val" => y_val,
    )
    if occursin("boehm", experimental_setting["problem_name"])
        test_res = Dict(
            "t_test" => missing,
            "prediction_hidden_test" => missing,
            "prediction_obs_test" => missing,
            "hidden_test" => missing,
            "obs_test" => missing,
        )
    else
        test_res = Dict(
            "t_test" => test_data.time,
            "prediction_hidden_test" => pred_test,
            "prediction_obs_test" => pred_obs_test,
            "hidden_test" => Matrix(test_data[:, 2:end])',
            "obs_test" => Matrix(test_data[:, Symbol.(OBSERVABLES_IDS)])',
        )
    end

    return train_res, val_res, test_res
end

function evaluate_one_array(ude_nr, result_path, experimental_settings)
    array_runtime_and_epoch_stats = get_epochs_and_runtime(ude_nr, result_path)

    experimental_setting =
        experimental_settings[experimental_settings.ude_nr.==ude_nr, :][1, :]
    res_train, res_val, res_test =
        predict_train_validation_test(experimental_setting, result_path)

    # check whether shapes match
    val_pred_success =
        size(res_val["prediction_hidden_val"])[2] == size(res_val["t_val"])[1]
    if occursin("boehm", experimental_setting["problem_name"])
        test_pred_success = false
        test_pred_defined = false
    else
        test_pred_success =
            size(res_test["prediction_hidden_test"])[2] == size(res_test["t_test"])[1]
        test_pred_defined = true
    end

    # load sigma information
    ude_nr = experimental_setting["ude_nr"]
    # set up UDE
    nn_model, _, _ = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)
    noise_parameter_bounds = get_noise_parameters_with_bounds(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
    )
    sigma = [
        inverse_transform(
            p_opt[parameter_name];
            lb = noise_parameter_bounds[parameter_name]["lowerBound"],
            ub = noise_parameter_bounds[parameter_name]["upperBound"],
            par_transform = noise_parameter_bounds[parameter_name]["parameter_transform"],
        ) for parameter_name in NOISE_PARAMETER_IDS
    ]


    metric_stats = DataFrame(array_runtime_and_epoch_stats)

    # MSE for training, validation and test
    # MSE for hidden states
    # MSE for observable states
    metric_stats[!, "mse_obs_train"] =
        [MSE(res_train["obs_train"], res_train["prediction_obs_train"])]
    if val_pred_success
        metric_stats[!, "mse_obs_val"] =
            [MSE(res_val["obs_val"], res_val["prediction_obs_val"])]
    else
        metric_stats[!, "mse_obs_val"] = [NaN]
    end

    if test_pred_success
        metric_stats[!, "mse_obs_test"] =
            [MSE(res_test["obs_test"], res_test["prediction_obs_test"])]
        metric_stats[!, "mse_hidden_test"] =
            [MSE(res_test["hidden_test"], res_test["prediction_hidden_test"])]
    else
        metric_stats[!, "mse_obs_test"] = [NaN]
        metric_stats[!, "mse_hidden_test"] = [NaN]
    end


    # nMSE for training, validation and test
    # nMSE for hidden states
    # nMSE for observable states
    metric_stats[!, "nmse_obs_train"] =
        [nMSE(res_train["obs_train"], res_train["prediction_obs_train"])]
    if val_pred_success
        metric_stats[!, "nmse_obs_val"] =
            [nMSE(res_val["obs_val"], res_val["prediction_obs_val"])]
        metric_stats[!, "nmse_obs_trainval"] = [
            nMSE(
                hcat(res_train["obs_train"], res_val["obs_val"]),
                hcat(res_train["prediction_obs_train"], res_val["prediction_obs_val"]),
            ),
        ]
        if test_pred_defined
            # hidden ids 
            train_ids = [
                i for i = 1:length(res_test["t_test"]) if
                res_test["t_test"][i] in res_train["t_train"]
            ]
            val_ids = [
                i for i = 1:length(res_test["t_test"]) if
                res_test["t_test"][i] in res_val["t_val"]
            ]
            metric_stats[!, "nmse_hidden_val"] = [
                nMSE(res_test["hidden_test"][:, val_ids], res_val["prediction_hidden_val"]),
            ]
            metric_stats[!, "nmse_hidden_train"] = [
                nMSE(
                    res_test["hidden_test"][:, train_ids],
                    res_train["prediction_hidden_train"],
                ),
            ]
            metric_stats[!, "nmse_hidden_trainval"] = [
                nMSE(
                    res_test["hidden_test"][:, vcat(train_ids, val_ids)],
                    hcat(
                        res_train["prediction_hidden_train"],
                        res_val["prediction_hidden_val"],
                    ),
                ),
            ]
        end
    else
        metric_stats[!, "nmse_obs_val"] = [NaN]
        metric_stats[!, "nmse_obs_trainval"] = [NaN]
    end
    if test_pred_success
        metric_stats[!, "nmse_obs_test"] =
            [nMSE(res_test["obs_test"], res_test["prediction_obs_test"])]
        metric_stats[!, "nmse_hidden_test"] =
            [nMSE(res_test["hidden_test"], res_test["prediction_hidden_test"])]
    else
        metric_stats[!, "nmse_obs_test"] = [NaN]
        metric_stats[!, "nmse_hidden_test"] = [NaN]
    end

    # negLL for training, validation and test
    # negLL for observable states
    metric_stats[!, "negLL_obs_train"] =
        [gaussian_negLL(res_train["obs_train"], res_train["prediction_obs_train"], sigma)]
    metric_stats[!, "mae_obs_train"] =
        [MAE(res_train["obs_train"], res_train["prediction_obs_train"])]
    metric_stats[!, "nmae_obs_train"] =
        [nMAE(res_train["obs_train"], res_train["prediction_obs_train"])]
    if val_pred_success
        metric_stats[!, "negLL_obs_val"] =
            [gaussian_negLL(res_val["obs_val"], res_val["prediction_obs_val"], sigma)]
        metric_stats[!, "negLL_obs_trainval"] = [
            gaussian_negLL(
                hcat(res_train["obs_train"], res_val["obs_val"]),
                hcat(res_train["prediction_obs_train"], res_val["prediction_obs_val"]),
                sigma,
            ),
        ]
        metric_stats[!, "mae_obs_val"] =
            [MAE(res_val["obs_val"], res_val["prediction_obs_val"])]
        metric_stats[!, "mae_obs_trainval"] = [
            MAE(
                hcat(res_train["obs_train"], res_val["obs_val"]),
                hcat(res_train["prediction_obs_train"], res_val["prediction_obs_val"]),
            ),
        ]
        metric_stats[!, "nmae_obs_val"] =
            [nMAE(res_val["obs_val"], res_val["prediction_obs_val"])]
        metric_stats[!, "nmae_obs_trainval"] = [
            nMAE(
                hcat(res_train["obs_train"], res_val["obs_val"]),
                hcat(res_train["prediction_obs_train"], res_val["prediction_obs_val"]),
            ),
        ]
    else
        metric_stats[!, "negLL_obs_val"] = [NaN]
        metric_stats[!, "negLL_obs_trainval"] = [NaN]
        metric_stats[!, "mae_obs_val"] = [NaN]
        metric_stats[!, "mae_obs_trainval"] = [NaN]
        metric_stats[!, "nmae_obs_val"] = [NaN]
        metric_stats[!, "nmae_obs_trainval"] = [NaN]
    end
    if test_pred_success
        metric_stats[!, "negLL_obs_test"] =
            [gaussian_negLL(res_test["obs_test"], res_test["prediction_obs_test"], sigma)]
        metric_stats[!, "mae_obs_test"] =
            [MAE(res_test["obs_test"], res_test["prediction_obs_test"])]
        metric_stats[!, "nmae_obs_test"] =
            [nMAE(res_test["obs_test"], res_test["prediction_obs_test"])]
    else
        metric_stats[!, "negLL_obs_test"] = [NaN]
        metric_stats[!, "mae_obs_test"] = [NaN]
        metric_stats[!, "nmae_obs_test"] = [NaN]
    end

    # add array_information
    metric_stats[!, "ude_nr"] = [ude_nr]

    for (i, ID) in enumerate(NOISE_PARAMETER_IDS)
        metric_stats[!, "$(ID)"] = [abs(sigma[i])]
    end

    return metric_stats
end

if length(unique(experimental_settings.problem_name)) > 1
    error("This script is only designed for one problem_name")
end
include(joinpath(pwd(), "4_ude/$(experimental_settings[1,"problem_name"]).jl"))
metric_stats_df = DataFrame()
for ude_nr in experimental_settings.ude_nr
    # println(ude_nr)
    try
        stats_sub = evaluate_one_array(ude_nr, result_path, experimental_settings)
        global metric_stats_df = vcat(metric_stats_df, stats_sub)
    catch
        println("Error in array $ude_nr")
    end
end

experiment_summary = outerjoin(experimental_settings, metric_stats_df, on = :ude_nr)
if split_summmary
    write(
        joinpath(result_path, "../experiment_summary_array_$(array_nr).csv"),
        experiment_summary,
    )
else
    write(joinpath(result_path, "../experiment_summary.csv"), experiment_summary)
end

include("../5_optimisation/utils.jl")
include("../2_measurements/load_data.jl")
using DataFrames: outerjoin
using CSV: write
using Plots

ude_nr = 83659

result_path = joinpath("6_evaluation", "basic_UDE_comparison", "results")

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

function ude_dynamics!(du, u, p::AbstractVector{T}, t, nn_model, st) where {T}
    J_0, k_1, k_2, k_3, k_4, k_6, k_ex, kappa, K_1, N, A, phi = p[1:12]

    S_1, S_2, S_3, S_4, N_2, A_3, S_4_ex = @inbounds u[1:7]

    h_1 = 1 + (A_3 / K_1)^4

    # mechanistic terms
    du_mech = zeros(T, 7)
    du_mech[1] = J_0 - k_1 * S_1 * A_3 / h_1
    du_mech[2] = 2 * (k_1 * S_1 * A_3) / h_1 - k_2 * S_2 * (N - N_2) - k_6 * S_2 * N_2
    du_mech[3] = k_2 * S_2 * (N - N_2) - k_3 * S_3 * (A - A_3)
    du_mech[4] = k_3 * S_3 * (A - A_3) - k_4 * S_4 * N_2 - kappa * (S_4 - S_4_ex)
    du_mech[5] = k_2 * S_2 * (N - N_2) - k_4 * S_4 * N_2 - k_6 * S_2 * N_2
    du_mech[6] = -2 * (k_1 * S_1 * A_3) / h_1 + 2 * k_3 * S_3 * (A - A_3) # - k_5*A_3 
    du_mech[7] = phi * kappa * (S_4 - S_4_ex) - k_ex * S_4_ex

    # NN component
    du_nn_full = zeros(7)
    du_nn = nn_model(u, p.nn, st)[1]
    # zero contribution to all states but state_idx_nn_out
    du_nn_full = zeros(T, 7)
    du_nn_full[[6]] = du_nn

    # combined dynamics
    for i = 1:7
        du[i] = du_mech[i] + du_nn_full[i]
    end
end;

"""
Mapping hidden states to observables
"""
function observable_mapping(state, p)
    NADH = @inbounds state[5, :]
    ATP = @inbounds state[6, :]

    return vcat(transpose(NADH), transpose(ATP))
end

function predict_train_validation_test(ude_nr, result_path)
    # define the neural network according to the experimental setting of a specified UDE
    # however: turn of input normalization
    summary_df =
        read("5_optimisation/2024_07_22_Ruoff_Grid/experiment_summary.csv", DataFrame)
    experimental_setting = summary_df[summary_df.ude_nr.==ude_nr, :][1, :]
    nn_model = define_nn_model(
        7,
        1,
        experimental_setting["act_fct"],
        experimental_setting["hidden_layers"],
        experimental_setting["hidden_neurons"],
        false,
    )

    IC = [1.187, 0.193, 0.05, 0.115, 0.077, 2.475, 0.077]
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)

    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, nn_model, st)

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

    pred_obs_train = observable_mapping(pred_train, p_opt)
    pred_obs_val = observable_mapping(pred_val, p_opt)
    if !occursin("boehm", experimental_setting["problem_name"])
        pred_obs_test = observable_mapping(pred_test, p_opt)
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


function evaluate_one_array(ude_nr, result_path)
    array_runtime_and_epoch_stats = get_epochs_and_runtime(ude_nr, result_path)

    res_train, res_val, res_test = predict_train_validation_test(ude_nr, result_path)

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
        metric_stats[!, "mae_obs_val"] = [NaN]
        metric_stats[!, "mae_obs_trainval"] = [NaN]
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
    metric_stats[!, "mae_obs_train"] =
        [MAE(res_train["obs_train"], res_train["prediction_obs_train"])]
    if val_pred_success
        metric_stats[!, "mae_obs_val"] =
            [MAE(res_val["obs_val"], res_val["prediction_obs_val"])]
        metric_stats[!, "mae_obs_trainval"] = [
            MAE(
                hcat(res_train["obs_train"], res_val["obs_val"]),
                hcat(res_train["prediction_obs_train"], res_val["prediction_obs_val"]),
            ),
        ]
    else
        metric_stats[!, "mae_obs_val"] = [NaN]
        metric_stats[!, "mae_obs_trainval"] = [NaN]
    end
    if test_pred_success
        metric_stats[!, "mae_obs_test"] =
            [MAE(res_test["obs_test"], res_test["prediction_obs_test"])]
    else
        metric_stats[!, "mae_obs_test"] = [NaN]
    end

    # add array_information
    metric_stats[!, "ude_nr"] = [ude_nr]

    return metric_stats
end

experimental_settings =
    read("5_optimisation/2024_07_22_Ruoff_Grid/experiment_summary.csv", DataFrame);
metric_stats_df = DataFrame()
for ude_nr in [83659]
    # println(ude_nr)
    try
        stats_sub = evaluate_one_array(ude_nr, result_path)
        write(joinpath(result_path, "experiment_summary_$(ude_nr).csv"), stats_sub)
    catch
        println("Error in array $ude_nr")
    end
end


function plot_ruoff_observed(res_train, res_val, res_test)
    p1 = plot(
        res_test["t_test"],
        res_test["obs_test"][1, :],
        label = "reference",
        xlabel = "time",
        ylabel = "N_2",
        title = "N_2",
    )
    plot!(res_test["t_test"], res_test["prediction_obs_test"][1, :], label = "prediction")
    scatter!(res_train["t_train"], res_train["obs_train"][1, :], label = "train")
    scatter!(res_val["t_val"], res_val["obs_val"][1, :], label = "validation")

    p2 = plot(
        res_test["t_test"],
        res_test["obs_test"][2, :],
        label = "reference",
        xlabel = "time",
        ylabel = "A3",
        title = "A3",
    )
    plot!(res_test["t_test"], res_test["prediction_obs_test"][2, :], label = "prediction")
    scatter!(res_train["t_train"], res_train["obs_train"][2, :], label = "train")
    scatter!(res_val["t_val"], res_val["obs_val"][2, :], label = "val")

    p = plot(p1, p2, layout = (1, 2), legend = (true, :bottomright))
    return p
end

function plot_ruoff_hidden(res_test)
    SPECIES_IDS = ("S_1", "S_2", "S_3", "S_4", "N_2", "A_3", "S_4_ex")
    ps = []
    for i = 1:7
        p1 = plot(res_test["t_test"], res_test["hidden_test"][i, :], label = "reference")
        plot!(
            res_test["t_test"],
            res_test["prediction_hidden_test"][i, :],
            label = "prediction",
        )
        push!(ps, p1)
    end

    p = plot(
        ps...,
        layout = 7,
        legend = false,
        size = (2000, 500),
        title = reshape([SPECIES_IDS...], 1, 7),
    )
    return p
end

res_train, res_val, res_test = predict_train_validation_test(ude_nr, result_path)
p_obs = plot_ruoff_observed(res_train, res_val, res_test)
p_hidden = plot_ruoff_hidden(res_test)

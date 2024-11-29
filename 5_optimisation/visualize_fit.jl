using Plots

ude_nr = 1016
experimental_setting = experimental_settings[experimental_settings.ude_nr.==ude_nr, :][1, :]
res_train, res_val, res_test =
    predict_train_validation_test(experimental_setting, result_path)

function plot_boehm_observed(ude_nr)
    nn_model, ude_dynamics!, observable_mapping = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)
    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)
    IC, IC_ids = get_IC_or_IC_IDS(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting["sparsity"],
        p_opt,
    )
    pred =
        predict_hidden_states_given_IC(0:1:240, p_opt, st, IC, true, dynamics!, KenCarp4())
    pred_obs = observable_mapping(pred, p_opt, st)

    p1 = plot(
        0:1:240,
        pred_obs[1, :],
        label = "pSTAT5A_rel",
        xlabel = "time",
        ylabel = "pSTAT5A_rel",
        title = "pSTAT5A_rel",
    )
    scatter!(res_train["t_train"], res_train["obs_train"][1, :], label = "train")
    scatter!(res_val["t_val"], res_val["obs_val"][1, :], label = "validation")
    p2 = plot(
        0:1:240,
        pred_obs[2, :],
        label = "pSTAT5B_rel",
        xlabel = "time",
        ylabel = "pSTAT5B_rel",
        title = "pSTAT5B_rel",
    )
    scatter!(res_train["t_train"], res_train["obs_train"][2, :], label = "train")
    scatter!(res_val["t_val"], res_val["obs_val"][2, :], label = "validation")
    p3 = plot(
        0:1:240,
        pred_obs[3, :],
        label = "rSTAT5A_rel",
        xlabel = "time",
        ylabel = "rSTAT5A_rel",
        title = "rSTAT5A_rel",
    )
    scatter!(res_train["t_train"], res_train["obs_train"][3, :], label = "train")
    scatter!(res_val["t_val"], res_val["obs_val"][3, :], label = "validation")
    p = plot(p1, p2, p3, layout = (1, 3), legend = (true, :bottomright), size = (1600, 400))
    return p
end

plot_boehm_observed(ude_nr)


function plot_boehm_hidden(ude_nr)
    nn_model, ude_dynamics!, observable_mapping = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)
    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)
    IC, IC_ids = get_IC_or_IC_IDS(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting["sparsity"],
        p_opt,
    )
    pred =
        predict_hidden_states_given_IC(0:1:240, p_opt, st, IC, true, dynamics!, KenCarp4())

    p1 = plot(
        0:1:240,
        pred[1, :],
        label = "STAT5A",
        xlabel = "time",
        ylabel = "STAT5A",
        title = "STAT5A",
    )
    p2 = plot(
        0:1:240,
        pred[2, :],
        label = "STAT5B",
        xlabel = "time",
        ylabel = "STAT5B",
        title = "STAT5B",
    )
    p3 = plot(
        0:1:240,
        pred[3, :],
        label = "pApA",
        xlabel = "time",
        ylabel = "pApA",
        title = "pApA",
    )
    p4 = plot(
        0:1:240,
        pred[4, :],
        label = "pApB",
        xlabel = "time",
        ylabel = "pApB",
        title = "pApB",
    )
    p5 = plot(
        0:1:240,
        pred[5, :],
        label = "pBpB",
        xlabel = "time",
        ylabel = "pBpB",
        title = "pBpB",
    )
    p6 = plot(
        0:1:240,
        pred[6, :],
        label = "nucpApA",
        xlabel = "time",
        ylabel = "nucpApA",
        title = "nucpApA",
    )
    p7 = plot(
        0:1:240,
        pred[5, :],
        label = "nucpApB",
        xlabel = "time",
        ylabel = "nucpApB",
        title = "nucpApB",
    )
    p8 = plot(
        0:1:240,
        pred[6, :],
        label = "nucpBpB",
        xlabel = "time",
        ylabel = "nucpBpB",
        title = "nucpBpB",
    )
    p = plot(
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        layout = (2, 4),
        legend = (true, :bottomright),
        size = (2000, 800),
    )
    return p
end

plot_boehm_hidden(ude_nr)



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

p_obs = plot_ruoff_observed(res_train, res_val, res_test)
p_hidden = plot_ruoff_hidden(res_test)

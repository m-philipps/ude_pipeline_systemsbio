using DataFrames: DataFrame
using CSV: write, read
# using ChainRulesCore: @ignore_derivatives
using Statistics: mean
using Optimization: OptimizationFunction, OptimizationProblem, solve, AutoZygote
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using Optim: LineSearches
using Zygote # for AD
using SciMLSensitivity # reverse-mode AD
using ForwardDiff, DifferentialEquations # for AD
using JLD2: save

include("../4_ude/ude.jl")

# -----------------------------------------
#       Experimental Setting
# -----------------------------------------

function create_overview_dataframe(experiment_name, hps_from_file)
    include("$(experiment_name)/experiment_definition.jl")

    # combine problems and noise levels to be investigated
    noise_sparsity_problems = ()
    for problem in exp_problems
        noise_sparsity_problems = collect(
            Iterators.product((problem,), exp_noise_levels[problem], exp_sparsity[problem]),
        )
        # flatten
        noise_sparsity_problems = [noise_sparsity_problems[1, :, :]...]
    end

    if hps_from_file
        experiments = DataFrame()
        hp_settings = read(hps_file_path, DataFrame)
        for (problem_name, noise, sparsity) in noise_sparsity_problems
            # for each noise-sparsity problem, we want to explore all HP of the hps_file
            sub_experiments = copy(hp_settings)
            sub_experiments[!, "problem_name"] .= problem_name
            sub_experiments[!, "noise_level"] .= noise
            sub_experiments[!, "sparsity"] .= sparsity
            if startpoint_method == "preopt"
                sub_experiments[!, "startpoint_id"] .= 1
            else
                sub_experiments[!, "startpoint_id"] = 1:size(sub_experiments, 1)
            end
            experiments = vcat(experiments, sub_experiments)
        end
        experiments[!, "ude_nr"] = 1:size(experiments, 1)
        experiments[!, "reg"] .= reg_method
        experiments[!, "startpoint_method"] .= startpoint_method
        experiments[!, "max_epochs_adam"] .= epochs[1]
        experiments[!, "max_epochs_bfgs"] .= epochs[2]
        experiments = experiments[
            !,
            [
                "ude_nr",
                "problem_name",
                "noise_level",
                "sparsity",
                "hidden_layers",
                "hidden_neurons",
                "act_fct",
                "nn_input_normalization",
                "startpoint_method",
                "startpoint_id",
                "reg",
                "λ_reg",
                "lr_adam",
                "max_epochs_adam",
                "max_epochs_bfgs",
            ],
        ]
        return experiments
    else

        # otherwise, based on the old grid set-up
        # combine startpoint information
        startpoints = ()
        for method in keys(exp_startpoints)
            startpoints =
                collect(Iterators.product((method,), exp_startpoints[method]))[1, :]
        end

        # combine above iterators with other hyperparameters
        experiments = collect(
            Iterators.product(
                noise_sparsity_problems,
                exp_hidden_layers,
                exp_hidden_neurons,
                exp_act_fct,
                exp_input_normalization,
                startpoints,
                exp_reg,
                exp_λ_reg,
                exp_lr_adam,
                epochs[1],
                epochs[2],
            ),
        )
        experiments = DataFrame(
            experiments,
            [
                :noise_sparsity_problems,
                :hidden_layers,
                :hidden_neurons,
                :act_fct,
                :nn_input_normalization,
                :startpoints,
                :reg,
                :λ_reg,
                :lr_adam,
                :max_epochs_adam,
                :max_epochs_bfgs,
            ],
        )
        # tidy dataframe
        experiments[!, "ude_nr"] = 1:size(experiments, 1)
        experiments[!, "problem_name"] =
            [el[1] for el in experiments[!, "noise_sparsity_problems"]]
        experiments[!, "noise_level"] =
            [el[2] for el in experiments[!, "noise_sparsity_problems"]]
        experiments[!, "sparsity"] =
            [el[3] for el in experiments[!, "noise_sparsity_problems"]]
        experiments[!, "startpoint_method"] =
            [el[1] for el in experiments[!, "startpoints"]]
        experiments[!, "startpoint_id"] = [el[2] for el in experiments[!, "startpoints"]]
        experiments = experiments[
            !,
            [
                "ude_nr",
                "problem_name",
                "noise_level",
                "sparsity",
                "hidden_layers",
                "hidden_neurons",
                "act_fct",
                "nn_input_normalization",
                "startpoint_method",
                "startpoint_id",
                "reg",
                "λ_reg",
                "lr_adam",
                "max_epochs_adam",
                "max_epochs_bfgs",
            ],
        ]

        return experiments
    end
end

function store_overview_dataframe(experiment_name, df)
    write("5_optimisation/$(experiment_name)/experiment_overview.csv", df)
end

function load_overview_dataframe(experiment_name)
    read("5_optimisation/$(experiment_name)/experiment_overview.csv", DataFrame)
end


# -----------------------------------------
#       Metrics
# -----------------------------------------

function gaussian_negLL(pred, obs, sigma)
    sum(
        1 / 2 * (
            size(obs)[2] * (log.(sigma .^ 2)) .+
            sum(abs2, pred .- obs, dims = 2) ./ (sigma .^ 2)
        ),
    )
end

function MSE(pred, obs)
    mean(abs2, pred .- obs)
end

function MAE(pred, obs)
    mean(abs, pred .- obs)
end

function nMAE(pred, obs)
    # normalization based on the mean value of the data 
    μ = mean(obs, dims = 2)
    mean(abs, (pred .- obs) ./ μ)
end

function nMSE(pred, obs)
    # normalization based on the mean value of the data 
    μ = mean(obs, dims = 2)
    mean(abs2, (pred .- obs) ./ μ)
end

function L2_norm(θ)
    convert(eltype(θ), mean(collect(θ) .^ 2))
end

function L1_norm(θ)
    convert(eltype(θ), mean(abs.(collect(θ))))
end


# -----------------------------------------
#       training
# -----------------------------------------

function base_loss(
    θ,
    st,
    t,
    y_obs,
    IC,
    IC_ids,
    t_includes_t_IC,
    observable_mapping,
    noise_parameter_bounds,
    noise_parameter_ids,
    dynamics!,
    solver,
)
    X̂ = missing
    if ismissing(IC)
        X̂ = predict_hidden_states(t, θ, st, IC_ids, t_includes_t_IC, dynamics!, solver)
    else
        X̂ =
            predict_hidden_states_given_IC(t, θ, st, IC, t_includes_t_IC, dynamics!, solver)
    end

    if size(X̂)[2] == size(y_obs)[2]
        pred = observable_mapping(X̂, θ, st)
        sigma = [
            inverse_transform(
                θ[parameter_name];
                lb = noise_parameter_bounds[parameter_name]["lowerBound"],
                ub = noise_parameter_bounds[parameter_name]["upperBound"],
                par_transform = noise_parameter_bounds[parameter_name]["parameter_transform"],
            ) for parameter_name in noise_parameter_ids
        ]

        # loss = sum_s(n_t * log(sigma) + 1/sigma² * sum_t (pred-true)²)
        # normalize w.r.t. n_t
        nll_value = sum(
            1 / 2 * (
                size(y_obs)[2] * (log.(sigma .^ 2)) .+
                sum(abs2, pred .- y_obs, dims = 2) ./ (sigma .^ 2)
            ),
        )
        return convert(eltype(θ), nll_value / prod(size(y_obs)))
    else
        return convert(eltype(θ), Inf)
    end
end;

function get_IC_or_IC_IDS(problem_name, noise_level, sparsity, θ)
    # if IC is estimated (Ruoff)
    # return missing, get_initial_condition_parameter_ids(θ)
    # if IC is not estimated (Ruoff)
    if problem_name == "ruoff_atp_consumption"
        return SPECIES_IC, missing
    elseif problem_name in [
        "boehm_papa_export_kinetic",
        "boehm_observable_ab_ratio",
        "boehm_papb_differential",
    ]
        return SPECIES_IC, missing
    elseif problem_name == "boehm_export_augmented"
        return vcat(SPECIES_IC, [0.0, 0.0, 0.0]), missing
    end
end

function train_ude(experimental_setting, result_path)
    # result path 
    if !isdir(result_path)
        mkpath(result_path)
    end
    if occursin("ruoff", experimental_setting["problem_name"])
        solver = Tsit5()
    else
        solver = KenCarp4()
    end
    # load training and validation data
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

    # define UDE and its initial parameters
    nn_model, ude_dynamics!, observable_mapping = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    # ps, st = load_test_parameter_values_and_state_ruoff(5, nn_model, "tanh_bounds")
    p_init, st = load_parameter_startpoints_and_state(experimental_setting, nn_model)
    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)

    # define loss functions
    IC, IC_ids = get_IC_or_IC_IDS(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting["sparsity"],
        p_init,
    )
    noise_parameter_bounds = get_noise_parameters_with_bounds(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
    )

    # mean nll (up to a constant factor)
    train_nll_fct(θ) = base_loss(
        θ,
        st,
        t_train,
        y_train,
        IC,
        IC_ids,
        true,
        observable_mapping,
        noise_parameter_bounds,
        NOISE_PARAMETER_IDS,
        dynamics!,
        solver,
    )
    val_nll_fct(θ) = base_loss(
        θ,
        st,
        t_val,
        y_val,
        IC,
        IC_ids,
        false,
        observable_mapping,
        noise_parameter_bounds,
        NOISE_PARAMETER_IDS,
        dynamics!,
        solver,
    )

    train_loss_fct(θ) = train_nll_fct(θ) + experimental_setting["λ_reg"] * L2_norm(θ)
    val_loss_fct(θ) = val_nll_fct(θ) + experimental_setting["λ_reg"] * L2_norm(θ)

    train_losses = Float32[]
    val_losses = Float32[Inf]
    # callback to track losses and optimal parameter values
    callback = function (state, l)
        push!(train_losses, l)
        val_loss = val_loss_fct(state.u)
        if val_loss <= minimum(val_losses)
            save(
                joinpath(result_path, "p_opt.jld2"),
                "p_opt",
                state.u,
                "epoch",
                length(train_losses),
            )
        end
        push!(val_losses, val_loss)
        return false
    end

    # First train with ADAM for better convergence -> move the parameters into a
    # favourable starting positing for BFGS
    adtype = AutoZygote()
    optf = OptimizationFunction((θ, p) -> train_loss_fct(θ), adtype) # θ: model parameters, p: hyperparameters
    optprob = OptimizationProblem(optf, p_init)
    res1 = solve(
        optprob,
        Adam(experimental_setting["lr_adam"]),
        callback = callback,
        maxiters = experimental_setting["max_epochs_adam"],
    )
    # then continue with BFGS
    optprob2 = OptimizationProblem(optf, res1.minimizer)
    res2 = solve(
        optprob2,
        BFGS(linesearch = LineSearches.BackTracking()),
        callback = callback,
        maxiters = experimental_setting["max_epochs_bfgs"],
        time_limit = 129600,
    ) # 1.5days # set upper time limit of 1day = 24h = 86 400s

    # save training results
    save(
        joinpath(result_path, "loss_curves.jld2"),
        "training_loss",
        train_losses,
        "validation_losses",
        val_losses[2:end],
    )
    save(
        joinpath(result_path, "training_times_and_epochs.jld2"),
        "Adam_time_in_s",
        res1.stats.time,
        "Adam_iterations",
        res1.stats.iterations,
        "BFGS_time_in_s",
        res2.stats.time,
        "BFGS_iterations",
        res2.stats.iterations,
    )
end

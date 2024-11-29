"""
Implement a single start UDE training with

1. no parameter bounds
2. MSE instead of a neg LL as objective function 
3. no input normalization
4. no early stopping
"""

# imports
include("../4_ude/ude.jl")
include("../2_measurements/load_data.jl")
using Statistics: mean
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

# helper functions
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

function L2_norm(θ)
    convert(eltype(θ), mean(collect(θ) .^ 2))
end


"""
Mapping hidden states to observables
"""
function observable_mapping(state, p)
    NADH = @inbounds state[5, :]
    ATP = @inbounds state[6, :]

    return vcat(transpose(NADH), transpose(ATP))
end

function load_parameter_startpoints_and_state(experimental_setting, nn_model)
    problem_name = experimental_setting["problem_name"]
    startpoint_method = experimental_setting["startpoint_method"]
    startpoint_id = experimental_setting["startpoint_id"]
    noise_level = experimental_setting["noise_level"]

    parameter_names = (
        "J_0",
        "k_1",
        "k_2",
        "k_3",
        "k_4",
        "k_6",
        "k_ex",
        "kappa",
        "K_1",
        "N",
        "A",
        "phi",
        # noise parameters
        "sd_N_2_obs",
        "sd_A_3_obs",
    )

    # read in parameters in the untransformed space
    if problem_name == "ruoff_atp_consumption"
        p_mech = read(
            "3_start_points/$(problem_name)/startpoints_$(startpoint_method)_noise_$(noise_level).csv",
            DataFrame,
        )[
            startpoint_id,
            :,
        ]
    end

    # use untransformed parameters
    p_mech_trans = NamedTuple{Symbol.(parameter_names)}([
        p_mech[par_name] for par_name in parameter_names
    ])

    # merge and return
    ps, st = Lux.setup(default_rng(), nn_model)
    p_nn = load(
        "3_start_points/$(problem_name)/fnn_$(experimental_setting["hidden_neurons"])_$(experimental_setting["hidden_layers"]).jld2",
        "p_nn_init",
    )
    ps = ComponentArray(ps)
    ps[1:end] = ComponentArray(p_nn)[1:end]
    ps = NamedTuple(ps)
    return ComponentArray(merge(p_mech_trans, (nn = ps,))), st
end

function base_loss(
    θ,
    st,
    t,
    y_obs,
    IC,
    t_includes_t_IC,
    observable_mapping,
    dynamics!,
    solver,
)
    X̂ = predict_hidden_states_given_IC(t, θ, st, IC, t_includes_t_IC, dynamics!, solver)

    if size(X̂)[2] == size(y_obs)[2]
        pred = observable_mapping(X̂, θ)
        return mean(abs2, pred .- y_obs)
    else
        return convert(eltype(θ), Inf)
    end
end;

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

    # define the neural network according to the experimental setting of a specified UDE
    # however: turn of input normalization
    nn_model = define_nn_model(
        7,
        1,
        experimental_setting["act_fct"],
        experimental_setting["hidden_layers"],
        experimental_setting["hidden_neurons"],
        false,
    )
    p_init, st = load_parameter_startpoints_and_state(experimental_setting, nn_model)

    dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, nn_model, st)

    # define loss functions
    IC = [1.187, 0.193, 0.05, 0.115, 0.077, 2.475, 0.077]

    # mean nll (up to a constant factor)
    train_nll_fct(θ) =
        base_loss(θ, st, t_train, y_train, IC, true, observable_mapping, dynamics!, solver)
    val_nll_fct(θ) =
        base_loss(θ, st, t_val, y_val, IC, false, observable_mapping, dynamics!, solver)

    train_loss_fct(θ) = train_nll_fct(θ) + experimental_setting["λ_reg"] * L2_norm(θ)
    val_loss_fct(θ) = val_nll_fct(θ) + experimental_setting["λ_reg"] * L2_norm(θ)

    train_losses = Float32[]
    val_losses = Float32[]
    # callback to track losses and optimal parameter values
    callback = function (state, l)
        push!(train_losses, l)
        val_loss = val_loss_fct(state.u)
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
    save(
        joinpath(result_path, "p_opt.jld2"),
        "p_opt",
        res1.minimizer,
        "epoch",
        length(train_losses),
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
    save(
        joinpath(result_path, "p_opt.jld2"),
        "p_opt",
        res2.minimizer,
        "epoch",
        length(train_losses),
    )

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

# conduct experiment
summary_df = read("5_optimisation/2024_07_22_Ruoff_Grid/experiment_summary.csv", DataFrame);

# ude_nr = 83659 # 123659 # 133659 # 83659

for ude_nr in [83659, 123659, 133659]
    experimental_setting = summary_df[summary_df.ude_nr.==ude_nr, :][1, :]
    result_path = "6_evaluation/basic_UDE_comparison/results/ude_$(ude_nr)"
    train_ude(experimental_setting, result_path)
end

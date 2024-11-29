include("../5_optimisation/utils.jl")
include("../2_measurements/load_data.jl")
using CSV
using DataFrames
using SymbolicRegression
using MLJ

experiment_name = "2024_07_22_Ruoff_Grid" # "2024_08_01_Boehm_papb_differential" # "2024_08_01_Boehm_papa_export_kinetic" # "2024_08_01_Boehm_observable_ab_ratio"
local_result_storage = false

result_path = "."
if local_result_storage
    result_path = joinpath("5_optimisation", experiment_name, "result")
else
    result_path = joinpath(
        "/storage/groups/hasenauer_lab/sym/5_optimisation/",
        experiment_name,
        "result",
    )
end

include(joinpath(pwd(), "4_ude/ruoff_atp_consumption.jl"))

function ude_prediction(experimental_setting, result_path, interpolate_t = false)
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
    if interpolate_t
        t = t_train[1]:1:t_train[end]
    else
        t = sort(vcat(t_train, t_val))
    end
    pred = predict_hidden_states_given_IC(t, p_opt, st, IC, true, dynamics!, solver)

    pred_obs = observable_mapping(pred, p_opt, st)

    return t, pred, pred_obs
end

NOISE = 5
SPARSITY = 200

df_summary = CSV.read(joinpath(result_path, "../experiment_summary.csv"), DataFrame)

settings_df = DataFrame(noise_level = [NOISE], sparsity = [SPARSITY])
df_summary = df_summary[
    all(eachrow(df_summary[:, names(settings_df)]) .== eachrow(settings_df), dims = 2),
    :,
]

df_summary[!, "reg_bin"] .= "-1"
df_summary[(df_summary[!, "λ_reg"].==0.0), "reg_bin"] .= "0"
df_summary[
    (df_summary[!, "λ_reg"].>=0.001).&(df_summary[!, "λ_reg"].<0.01),
    "reg_bin",
] .= "[1e-3, 1e-2["
df_summary[
    (df_summary[!, "λ_reg"].>=0.01).&(df_summary[!, "λ_reg"].<0.1),
    "reg_bin",
] .= "[1e-2, 1e-1["
df_summary[
    (df_summary[!, "λ_reg"].>=0.1).&(df_summary[!, "λ_reg"].<1.0),
    "reg_bin",
] .= "[1e-1, 1.0["
df_summary[
    (df_summary[!, "λ_reg"].>=1.0).&(df_summary[!, "λ_reg"].<=10.0),
    "reg_bin",
] .= "[1, 10]"

combine(groupby(df_summary, [:reg_bin]), nrow => :count)

# select best ude_nr for symbolic regression
# globally 
for reg_bin in ["0", "[1e-3, 1e-2[", "[1e-2, 1e-1[", "[1e-1, 1.0[", "[1, 10]"]

    df_summary_sub = df_summary[df_summary[!, "reg_bin"].==reg_bin, :] # |> x -> x[x.negLL_obs_trainval .== minimum(x[!,:negLL_obs_trainval]), :ude_nr]
    df_summary_sub = df_summary_sub[(!).(ismissing.(df_summary_sub.negLL_obs_trainval)), :]
    ude_nr = df_summary_sub[
        df_summary_sub[
            !,
            :negLL_obs_trainval,
        ].==minimum(df_summary_sub[!, :negLL_obs_trainval]),
        :ude_nr,
    ][1]
    println("Reg Bin: $reg_bin, UDE Nr: $(ude_nr)")

    # load overview of experiments
    # experimental_settings = CSV.read(joinpath(result_path, "../experiment_overview.csv"), DataFrame)
    # experimental_setting = experimental_settings[experimental_settings.ude_nr .== ude_nr, :][1,:]
    experimental_setting = df_summary[df_summary.ude_nr.==ude_nr, :][1, :]

    # create predictions and targets for symbolic regression
    interpolate_t = true
    t_res, pred_res, pred_obs_res =
        ude_prediction(experimental_setting, result_path, interpolate_t)

    nn_model, ude_dynamics!, observable_mapping = define_ude_model(
        experimental_setting["problem_name"],
        experimental_setting["noise_level"],
        experimental_setting,
    )
    p_opt, st = load_optimal_parameters_and_state(result_path, ude_nr, nn_model)

    X = pred_res
    nn_pred = nn_model(X, p_opt.nn, st)[1]
    y = nn_pred[1, :]

    X_y = DataFrame(transpose(X), [Symbol.(id) for id in SPECIES_IDS])
    X_y.y_nn = y

    if occursin("observable", experiment_name)
        model = SRRegressor(
            niterations = 1000,
            binary_operators = [+, -, *, /],
            maxsize = 35, # default is 19
            output_file = "6_evaluation/SR_results/hall_of_fame_$(experiment_name)_$(ude_nr)_interpolate_$(interpolate_t).csv",
        )
    else
        model = SRRegressor(
            niterations = 500,
            binary_operators = [+, -, *], # /
            output_file = "6_evaluation/SR_results/hall_of_fame_$(experiment_name)_$(ude_nr)_interpolate_$(interpolate_t).csv",
        )
    end

    mach = machine(model, X_y[:, [Symbol.(id) for id in SPECIES_IDS]], X_y.y_nn)
    fit!(mach)

    X_y.y_SR = predict(mach, X_y[:, [Symbol.(id) for id in SPECIES_IDS]])
    CSV.write(
        "6_evaluation/SR_results/predictions_$(experiment_name)_$(ude_nr)_interpolate_$(interpolate_t).csv",
        X_y,
    )


    r = report(mach)
    r.equations[r.best_idx]

    save(
        "6_evaluation/SR_results/report_$(experiment_name)_$(reg_bin)_$(ude_nr).jld2",
        "r",
        r,
    )
end

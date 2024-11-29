"""Simulate the n best runs by metric, from experiment_summary.csv."""

using ComponentArrays
using OrdinaryDiffEq: Tsit5, KenCarp4
using CSV, DataFrames, DelimitedFiles
using JSON: parsefile

# manual settings
CLUSTER = true;
problem_name = "boehm_export_augmented"
experiment_name::String = string("2024_08_27_B", problem_name[2:end])
metric::String = "nmae_obs_trainval"  # nmse_obs_test, negLL_obs_trainval
n_best::Int64 = 50

# load reference, simulation function, observable mapping
dir_pipeline = dirname(@__DIR__)
include(joinpath(dir_pipeline, "1_mechanistic_model", "reference_boehm.py"))
include(joinpath(dir_pipeline, "4_ude", "ude.jl"))
include(joinpath(dir_pipeline, "4_ude", "$problem_name.jl"))
include(joinpath(dir_pipeline, "5_optimisation", "utils.jl"))

problem_setting = parsefile(joinpath(dir_pipeline, "problems.json"))[problem_name]
if haskey(problem_setting, "augmentation") &&
   (problem_setting["augmentation"]["type"] == "species")
    species_ids = vcat(collect(SPECIES_IDS), problem_setting["augmentation"]["name"])
else
    species_ids = collect(SPECIES_IDS)
end


# set paths
if CLUSTER
    storage_dir = "/storage/groups/hasenauer_lab/sym"
    exp_output_dir = joinpath(storage_dir, "5_optimisation", experiment_name)
    sim_output_dir = joinpath(storage_dir, "6_evaluation", "simulation", experiment_name)
else
    dir_pipeline = dirname(@__DIR__)
    exp_output_dir = joinpath(dir_pipeline, "5_optimisation", experiment_name)
    sim_output_dir = joinpath(dir_pipeline, "6_evaluation", "simulation", experiment_name)
end


# read in experiment summary
fp = joinpath(exp_output_dir, "experiment_summary.csv")
summary = DataFrame(CSV.File(fp))
# subset to only successful experiments
summary = dropmissing(summary, NOISE_PARAMETER_IDS)
# sort by metric
sort!(summary, metric)
# simulate the n best
for row in eachrow(summary)[1:n_best]
    ude_nr = row["ude_nr"]
    println(ude_nr)
    # load UDE
    (nn_model, ude_dynamics!, observable_mapping) =
        define_ude_model(problem_name, row["noise_level"], row)
    # load IC
    ic, _ = get_IC_or_IC_IDS(problem_name, false, false, false)
    # load optimised parameter
    p_opt, st = load_optimal_parameters_and_state(
        joinpath(exp_output_dir, "result"),
        ude_nr,
        nn_model,
    )
    dynamics_opt!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)
    # prediction
    t = collect(T_START:0.1:T_END)
    sim = predict_hidden_states_given_IC(t, p_opt, st, ic, true, dynamics_opt!, KenCarp4())
    obs = observable_mapping_full(sim, p_opt, nn_model, st)
    # save
    df = DataFrame(
        hcat(t[1:size(sim)[2]], transpose(sim), transpose(obs)),
        vcat(["Time"], species_ids, collect(OBSERVABLES_IDS)),
    )
    fp_output = joinpath(sim_output_dir, "ude_$ude_nr")
    mkpath(fp_output)
    CSV.write(joinpath(fp_output, "simulation.csv"), df)
end

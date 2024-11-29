"""Simulate the n best runs by metric, for all individual experiment_summary_array_i.csv 
corresponding to each individual noise/sparsity setting."""

using ComponentArrays
using OrdinaryDiffEq: Tsit5, KenCarp4
using CSV, DataFrames, DelimitedFiles
using CategoricalArrays: cut

# load simulation function
include("../4_ude/ude.jl")

# load IC
include("../1_mechanistic_model/reference_ruoff.py")

# manual settings
CLUSTER = true;
problem_name = "ruoff_atp_consumption"

experiment_name::String = "2024_07_22_Ruoff_Grid"
metric::String = "nmae_obs_test"  # nmse_obs_test, negLL_obs_trainval
n_best::Int64 = 50

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

# add binned regularisation strength
regbins = [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
regbin_labels = ["0", "<0.01", "<0.1", "<1", "<10"]
summary.regbin = cut(summary.λ_reg, regbins, labels = regbin_labels)

# subset by dataset and regularisation
for (noise, n_datapoints, reg) in
    collect(Base.product(NOISE_PERCENTAGES, DATASET_SIZES, labels))
    println("\n", noise, " % noise,  # dp ", n_datapoints, ", reg. ", reg)
    settings_df =
        DataFrame(noise_level = [noise], sparsity = [n_datapoints], regbin = [reg])
    subdf = summary[
        all(eachrow(summary[:, names(settings_df)]) .== eachrow(settings_df), dims = 2),
        :,
    ]
    if size(subdf, 1) < 2
        continue
    end
    # sort by metric
    sort!(subdf, metric)
    # simulate the n best
    for row in eachrow(subdf)[1:n_best]
        ude_nr = row["ude_nr"]
        println(ude_nr)
        noise_level = row["noise_level"]
        # load UDE
        (nn_model, ude_dynamics!, observable_mapping) =
            define_ude_model(problem_name, noise_level, row)
        # load optimised parameter
        p_opt, st = load_optimal_parameters_and_state(
            joinpath(exp_output_dir, "result"),
            ude_nr,
            nn_model,
        )
        dynamics_opt!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)
        # prediction
        t = collect(0:0.1:TEST_ENDPOINT)
        sim = predict_hidden_states_given_IC(
            t,
            p_opt,
            st,
            SPECIES_IC,
            true,
            dynamics_opt!,
            Tsit5(),
        )
        # save
        df = DataFrame(
            hcat(t[1:size(sim)[2]], transpose(sim)),
            vcat(["Time"], collect(SPECIES_IDS)),
        )
        fp_output = joinpath(sim_output_dir, "ude_$ude_nr")
        mkpath(fp_output)
        CSV.write(joinpath(fp_output, "simulation.csv"), df)
    end
end

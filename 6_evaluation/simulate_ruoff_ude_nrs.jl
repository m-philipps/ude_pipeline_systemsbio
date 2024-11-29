using ComponentArrays
using OrdinaryDiffEq: Tsit5, KenCarp4
using CSV, DataFrames, DelimitedFiles

# load simulation function
include("../4_ude/ude.jl")

# load IC
include("../1_mechanistic_model/reference_ruoff.py")

# manual settings
CLUSTER = true;
problem_name = "ruoff_atp_consumption"
experiment_name::String = "2024_07_22_Ruoff_Grid"
# read in ude nrs
filepath = joinpath(@__DIR__, experiment_name, "ude_nrs_by_reg.txt")
ude_nrs_to_sim = readdlm(filepath)

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
summary = DataFrame(CSV.File(fp));
# subset to only successful experiments
summary = dropmissing(summary, NOISE_PARAMETER_IDS);


# simulate by given list
subdf = summary[in.(summary.ude_nr, Ref(ude_nrs_to_sim)), :]
for row in eachrow(subdf)
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

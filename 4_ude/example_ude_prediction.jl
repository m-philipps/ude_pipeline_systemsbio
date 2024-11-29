"""
Examplary pipeline to define a UDE model, load parameters and use these parameters to form predictions
"""

include("ude.jl")

experimental_setting = Dict(
    "problem_name" => "ruoff_atp_consumption",
    "sparsity" => 100,
    "noise_level" => 5,
    "act_fct" => "tanh",
    "hidden_layers" => 2,
    "hidden_neurons" => 3,
    "startpoint_method" => "lhs",
    "startpoint_id" => 20,
    "nn_input_normalization" => true,
)

nn_model, ude_dynamics!, observable_mapping = define_ude_model(
    experimental_setting["problem_name"],
    experimental_setting["noise_level"],
    experimental_setting,
)
# ps, st = load_test_parameter_values_and_state_ruoff(5, nn_model, "tanh_bounds")
ps, st = load_parameter_startpoints_and_state(experimental_setting, nn_model)

dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, st)

IC = [1.187, 0.193, 0.05, 0.115, 0.077, 2.475, 0.077]
t = 0.0:0.5:10.0

pred_hidden = predict_hidden_states_given_IC(t, ps, st, IC, true, dynamics!)

pred_obs = observable_mapping(pred_hidden, ps, st)

nn_model(IC, ps.nn, st)

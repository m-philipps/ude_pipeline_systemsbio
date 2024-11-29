using JSON: parsefile
using JLD2: load
using Lux
using Random: default_rng, seed!
using CSV: read
using DataFrames: DataFrame
using ComponentArrays: ComponentVector, ComponentArray
using OrdinaryDiffEq: ODEProblem, Tsit5, KenCarp4, solve

# TODO: check shift in parameter transform: still necessary?

"""
Given parameters in the untransformed space, transform them to the transformed space 

#### Arguments:
    par:        parameter value
    lb:         lower bound on parameter
    ub:         upper bound on parameter
    transform:  parameter transform used, must be one of
                - "tanh_bounds": tanh, shifted and scaled to fit lb and ub
                - "log": log transform without bounds
                - "identity": no transform
"""
function transform_par(par; lb::Float64, ub::Float64, par_transform::String)
    if par_transform == "tanh_bounds"
        # For numerical stability, shift the bounds slightly
        lb = max(0.0, lb * (1-1e-6))
        ub = ub * (1+1e-6)
        return atanh(2*(par-lb)/(ub-lb)-1) + 1.73
    elseif par_transform == "log"
        return log(par)
    elseif par_transform == "identity"
        return par
    end
end

"""
Given parameters in the transformed space, transform them back to the parameter's actual values 

#### Arguments:
    par:        parameter value
    lb:         lower bound on parameter
    ub:         upper bound on parameter
    transform:  parameter transform used, must be one of
                - "tanh_bounds": tanh, shifted and scaled to fit lb and ub
                - "log": log transform without bounds
                - "identity": no transform
"""
function inverse_transform(par; lb::Float64, ub::Float64, par_transform::String)
    if par_transform == "tanh_bounds"
        # For numerical stability, shift the bounds slightly
        lb = lb * (1-1e-6)
        ub = ub * (1+1e-6)
        return lb + (Lux.tanh(par-1.73) + 1)/2*(ub-lb)
    elseif par_transform == "log"
        return Lux.exp(par)
    elseif par_transform == "identity"
        return par
    end
end;


function load_parameter_startpoints_and_state(experimental_setting, nn_model)
    problem_name = experimental_setting["problem_name"]
    startpoint_method = experimental_setting["startpoint_method"]
    startpoint_id = experimental_setting["startpoint_id"]
    noise_level = experimental_setting["noise_level"]

    # read in parameters in the untransformed space
    if problem_name == "ruoff_atp_consumption"
        p_mech = read("3_start_points/$(problem_name)/startpoints_$(startpoint_method)_noise_$(noise_level).csv", DataFrame)[startpoint_id, :]
    elseif problem_name == "boehm_observable_ab_ratio"
        p_mech = read("3_start_points/boehm_observable_ab_ratio/startpoints_$(startpoint_method).csv", DataFrame)[startpoint_id, :]
    elseif problem_name == "boehm_export_augmented"
        p_mech = read("3_start_points/boehm_export_augmented/startpoints_$(startpoint_method).csv", DataFrame)[startpoint_id, :]
    elseif occursin("boehm", problem_name)
        p_mech = read("3_start_points/boehm_full/startpoints_$(startpoint_method).csv", DataFrame)[startpoint_id, :]
    end

    # transform parameters
    parameter_names, parameter_bounds = get_parameter_bounds(problem_name, noise_level)
        
    p_mech_trans = NamedTuple{Symbol.(parameter_names)}([transform_par(p_mech[par_name]; 
                                                    lb = parameter_bounds[par_name]["lowerBound"], 
                                                    ub = parameter_bounds[par_name]["upperBound"], 
                                                    par_transform = parameter_bounds[par_name]["parameter_transform"]) for par_name in parameter_names])
    
    # sample NN parameters randomly, if not already provided
    seed!(experimental_setting["ude_nr"])
    ps, st = Lux.setup(default_rng(), nn_model)
    if !(problem_name == "boehm_export_augmented")
        p_nn = load("3_start_points/$(problem_name)/fnn_$(experimental_setting["hidden_neurons"])_$(experimental_setting["hidden_layers"]).jld2", "p_nn_init")
        ps = ComponentArray(ps)
        ps[1:end] = ComponentArray(p_nn)[1:end]
        ps = NamedTuple(ps)
    end
    # merge and return
    return ComponentArray(merge(p_mech_trans, (nn=ps,))), st
end

function load_optimal_parameters_and_state(result_path, ude_nr, nn_model)
    ps, st = Lux.setup(default_rng(), nn_model)
    p_opt = load(joinpath(result_path, "ude_$(ude_nr)", "p_opt.jld2"), "p_opt")

    return p_opt, st
end

function load_test_parameter_values_and_state_ruoff(noise_level, nn_model, par_transform)
    p_mech = ComponentVector{Float64}(; J_0 = 2.5,
            k_1 = 100.,
            k_2 = 6.,
            k_3 = 16.,
            k_4 = 100.,
            k_6 = 12.,
            k_ex = 1.8,
            kappa = 13.,
            K_1 = 0.52,
            N = 1.,
            A = 4.,
            phi = 0.1,
            S_1_init = 1.187,
            S_2_init = 0.193,
            S_3_init = 0.05,
            S_4_init = 0.115,
            S_4_ex_init = 0.077,
            N_2_init = 2.475,
            A_3_init = 0.077,
            sd_N_2_obs = 0.01,
            sd_A_3_obs = 0.1,);
    
    parameter_names, parameter_bounds = get_parameter_bounds("ruoff_atp_consumption", noise_level)
    individual_parameter_transform = map_parameter_transform_to_name(par_transform, parameter_names)
    p_mech_trans = NamedTuple{Symbol.(parameter_names)}([transform_par(p_mech[par_name]; 
                                                    lb = parameter_bounds[par_name]["lowerBound"], 
                                                    ub = parameter_bounds[par_name]["upperBound"], 
                                                    par_transform=individual_parameter_transform[par_name]) for par_name in parameter_names])
    
    ps, st = Lux.setup(default_rng(), nn_model) 
    return ComponentArray(merge(p_mech_trans, (nn=ps,))), st
end

function get_parameter_bounds(problem_name, noise_level)
    if problem_name == "boehm_papb_differential_BaF3_Epo"
        problem_name = "boehm_papb_differential"
    end

    problem_definition = parsefile("problems.json")[problem_name]
    # Load parameter names
    if occursin("ruoff", problem_name)
        include(joinpath(pwd(),"1_mechanistic_model/reference_ruoff.py"))
    elseif occursin("boehm", problem_name)
        include(joinpath(pwd(),"1_mechanistic_model/reference_boehm.py"))
    end

    # Load information on parameter bounds for estimated parameters
    estimated_parameter_ids = [i for i in 1:length(problem_definition["mechanistic_parameters"]) if problem_definition["mechanistic_parameters"][i]==1]
    parameter_bounds = Dict()
    petab_parameter_file = ""
    
    if problem_name == "ruoff_atp_consumption"
        petab_parameter_file = read("1_mechanistic_model/Ruoff_BPC2003/parameters_noise_$(noise_level).tsv", DataFrame)
    elseif problem_name == "boehm_export_augmented"
        petab_parameter_file = read("1_mechanistic_model/Boehm_JProteomeRes2014/parameters_Boehm_JProteomeRes2014_augmented.tsv", DataFrame)
    elseif occursin("boehm", problem_name)
        petab_parameter_file = read("1_mechanistic_model/Boehm_JProteomeRes2014/parameters_Boehm_JProteomeRes2014.tsv", DataFrame)
    end
    
    if problem_name == "boehm_export_augmented"
        ordered_parameter_names = Tuple(vcat([PARAMETERS_IDS...], [PARAMETERS_IDS_AUGMENTATION...])[estimated_parameter_ids])
    else
        ordered_parameter_names = PARAMETERS_IDS[estimated_parameter_ids]
    end

    for parameter_id in ordered_parameter_names
        row = petab_parameter_file[petab_parameter_file.parameterId .== parameter_id, :]
        parameter_transform = row[1, "parameterScale"] == "log10" ? "tanh_bounds" : "identity"
        
        # include check for initial conditions (IC is not transformed when predicting)
        if occursin("_init", parameter_id) && parameter_transform != "identity"
            error("not implemented")
        end

        parameter_bounds[parameter_id] = Dict("lowerBound" => Float64(row[1,"lowerBound"]), "upperBound" => Float64(row[1,"upperBound"]), "parameter_transform" => parameter_transform)
    end

    return ordered_parameter_names, parameter_bounds
end

function define_nn_model(neurons_in, neurons_out, act_fct_name, hidden_layers, hidden_neurons, input_normalization)
    if act_fct_name == "tanh"
        act_fct = Lux.tanh
    elseif act_fct_name == "gelu"
        act_fct = Lux.gelu
    elseif act_fct_name == "relu"
        act_fct = Lux.relu
    elseif act_fct_name == "swish"
        act_fct = Lux.swish
    elseif act_fct_name == "rbf"
        act_fct = x -> Lux.exp.(-(x.^2))
    elseif act_fct_name == "identity"
        act_fct = x -> x
    end

    normalization_helper = x -> Lux.relu(x) .+ 1e-20

    # Construct a feedforward neural network
    first_layer = Dense(neurons_in, hidden_neurons, act_fct)
    intermediate_layers = [Dense(hidden_neurons, hidden_neurons, act_fct) for _ in 1:hidden_layers-1]
    last_layer = Dense(hidden_neurons, neurons_out, init_weight=Lux.zeros32)
    if Bool(input_normalization)
        nn_model = Lux.Chain(WrappedFunction(Base.Fix1(broadcast, normalization_helper)), WrappedFunction(Base.Fix1(broadcast, Lux.log)), first_layer, intermediate_layers..., last_layer)
    else
        nn_model = Lux.Chain(first_layer, intermediate_layers..., last_layer)
    end
    return nn_model
end

function define_ude_model(problem_name, noise_level, hyperparameters)
    # Define NN model
    if problem_name == "boehm_papb_differential_BaF3_Epo"
        problem_definition = parsefile("problems.json")["boehm_papb_differential"]
        neurons_in = sum(problem_definition["nn_input"]["ids"])
        neurons_out = sum(problem_definition["nn_output"]["ids"])
        neurons_in += 1
    else
        problem_definition = parsefile("problems.json")[problem_name]
        neurons_in = sum(problem_definition["nn_input"]["ids"])
        neurons_out = sum(problem_definition["nn_output"]["ids"])
    end
    nn_model = define_nn_model(neurons_in, neurons_out, hyperparameters["act_fct"], hyperparameters["hidden_layers"], hyperparameters["hidden_neurons"], hyperparameters["nn_input_normalization"])

    state_idx_nn_in = [i for i in 1:length(problem_definition["nn_input"]["ids"]) if problem_definition["nn_input"]["ids"][i]==1]
    state_idx_nn_out = [i for i in 1:length(problem_definition["nn_output"]["ids"]) if problem_definition["nn_output"]["ids"][i]==1]
    
    include(joinpath(pwd(),"4_ude/$(problem_name).jl"))
    # define the ude
    estimated_parameter_names, parameter_bounds = get_parameter_bounds(problem_name, noise_level)
    dynamic_parameter_names = get_dynamic_parameter_names(problem_name, estimated_parameter_names)
    dynamic_parameter_bounds = Dict("lowerBound" => [parameter_bounds[par_name]["lowerBound"] for par_name in dynamic_parameter_names],
                                    "upperBound" => [parameter_bounds[par_name]["upperBound"] for par_name in dynamic_parameter_names],
                                    "parameter_transform" => [parameter_bounds[par_name]["parameter_transform"] for par_name in dynamic_parameter_names])
    if occursin("ruoff", problem_name)
        # check if the ordering of dynamic parameters match what is expected in the Ruoff atp model
        @assert dynamic_parameter_names == ("J_0", "k_1", "k_2", "k_3", "k_4", "k_6", "k_ex", "kappa", "K_1", "N", "A", "phi") "Dynamic parameters do not match the Ruoff model"
    end    
    ude_dynamics!(du, u, p, t, st_nn) = ude_dynamics_full!(du, u, p, t, dynamic_parameter_bounds, nn_model, st_nn, (state_idx_nn_in, state_idx_nn_out))
    observable_mapping(state, p, st_nn) = observable_mapping_full(state, p, nn_model, st_nn)

    return nn_model, ude_dynamics!, observable_mapping
end

function get_initial_condition_parameter_ids(parameters)
    return [i for i in eachindex(keys(parameters)) if occursin("_init", String(keys(parameters)[i]))]
end

function get_dynamic_parameter_names(problem_name, estimated_parameter_names)
    if problem_name == "ruoff_atp_consumption"
        include(joinpath(pwd(),"1_mechanistic_model/reference_ruoff.py"))
    end

    return estimated_parameter_names[[i for i in eachindex(estimated_parameter_names) if ! (estimated_parameter_names[i] in NOISE_PARAMETER_IDS) ]]
end

function get_noise_parameters_with_bounds(problem_name, noise_level)
    if problem_name == "ruoff_atp_consumption"
        include(joinpath(pwd(),"1_mechanistic_model/reference_ruoff.py"))
    end
    _, parameter_bounds = get_parameter_bounds(problem_name, noise_level)
    return Dict(noise_parameter => parameter_bounds[noise_parameter] for noise_parameter in NOISE_PARAMETER_IDS)
end


"""
predict hidden state based on the given parameters (including parameters for the initial condition)
"""
function predict_hidden_states(t, parameters, st, IC_ids, t_includes_t_IC, dynamics!, solver)
    if !t_includes_t_IC
        t = vcat([0.0], t)
    end
    _prob = ODEProblem{true}(dynamics!, parameters[IC_ids], (t[1], t[end]), parameters)
    tmp_sol = solve(_prob, solver, saveat = t, verbose = false)
    if t_includes_t_IC
        return Array(tmp_sol)
    else
        return Array(tmp_sol)[:, 2:end]
    end
end

"""
predict hidden state based on the given parameters and a known initial condition
"""
function predict_hidden_states_given_IC(t, parameters, st, IC, t_includes_t_IC, dynamics!, solver)
    if !t_includes_t_IC
        t = vcat([0.0], t)
    end
    _prob = ODEProblem{true}(dynamics!, eltype(parameters).(IC), eltype(parameters).((t[1], t[end])), parameters)
    tmp_sol = solve(_prob, solver, saveat = t,verbose = false)
    if t_includes_t_IC
        return Array(tmp_sol)
    else
        return Array(tmp_sol)[:, 2:end]
    end
end
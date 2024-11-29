using JSON
using Lux
using Random
using JLD2

rng = Random.default_rng()
Random.seed!(rng, 2)

dir_pipeline = dirname(dirname(@__FILE__))

# load hp ref
hp_options = JSON.parsefile(joinpath(dirname(@__FILE__), "reference_ruoff.json"))
# load problems
problems = JSON.parsefile(joinpath(dir_pipeline, "problems.json"))

problem_name = "ruoff_atp_consumption"
problem = problems[problem_name]

n_input = sum(problem["nn_input"]["ids"])
n_output = sum(problem["nn_output"]["ids"])

# load hp grid
widths = hp_options["hp_options"]["fnn_width"]
depths = hp_options["hp_options"]["fnn_depth"]

settings =
    [(n_input, n_output, t[1], t[2]) for t in collect(Iterators.product(widths, depths))]

for (n_input, n_output, width, depth) in settings
    # construct a feedforward neural network
    first_layer = Dense(n_input, width)
    intermediate_layers = [Dense(width, width) for _ = 1:depth-1]
    last_layer = Dense(width, n_output, init_weight = Lux.zeros32)
    nn_model = Lux.Chain(first_layer, intermediate_layers..., last_layer)
    # extract parameters
    ps, st = Lux.setup(rng, nn_model)
    # save
    filename = string("fnn_", width, "_", depth, ".jld2")
    filepath = joinpath(dir_pipeline, "3_start_points", problem_name, filename)
    save(filepath, "p_nn_init", ps)
end

using JSON
using Lux
using Random
using JLD2

rng = Random.default_rng()
Random.seed!(rng, 2)

dir_pipeline = dirname(dirname(@__FILE__))
boehm_problem = "boehm_export_augmented"

# load hp ref
hp_options = JSON.parsefile(joinpath(dirname(@__FILE__), "reference_boehm.json"))
# hp_options = JSON.parsefile(joinpath(dirname(@__FILE__), "reference_$boehm_problem.json"))
n_starts = hp_options["n_starts"]

# load problems
problems = JSON.parsefile(joinpath(dir_pipeline, "problems.json"))

for problem_name in [boehm_problem]  # problems["boehm_ids"]
    problem = problems[problem_name]
    n_input = sum(problem["nn_input"]["ids"])
    n_output = sum(problem["nn_output"]["ids"])

    # load hp grid
    # widths = hp_options["hp_options"]["fnn_width"]
    # depths = hp_options["hp_options"]["fnn_depth"]
    # settings = [
    #     (
    #         n_input, n_output, t[1], t[2]
    #     ) for t in collect(Iterators.product(widths, depths))
    # ]
    # for (n_input, n_output, width, depth) in settings

    # one ANN size
    width = hp_options["hp_options"]["fnn_width"][1]
    depth = hp_options["hp_options"]["fnn_depth"][1]
    for i = 1:n_starts

        # construct a feedforward neural network
        first_layer = Dense(n_input, width)
        intermediate_layers = [Dense(width, width) for _ = 1:depth-1]
        last_layer = Dense(width, n_output, init_weight = Lux.zeros32)
        nn_model = Lux.Chain(first_layer, intermediate_layers..., last_layer)
        # extract parameters
        ps, st = Lux.setup(rng, nn_model)
        # save
        filename = string("fnn_", i, ".jld2")
        filepath = joinpath(
            dir_pipeline,
            "3_start_points",
            "boehm_export_augmented_preopt",
            "fnn",
            filename,
        )
        save(filepath, "p_nn_init", ps)
    end
end

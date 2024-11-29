using DataFrames: DataFrame, unstack
using CSV: read

function load_measurements(problem_name, sparsity, noise_level, data_subset)
    data = missing
    if occursin("ruoff", problem_name)
        data = read(joinpath("2_measurements", "Ruoff_synthetic_data", "data_$(sparsity)_noise_$(noise_level)_$(data_subset).tsv"), DataFrame)
    elseif occursin("boehm", problem_name)
        data = read(joinpath("1_mechanistic_model", "Boehm_JProteomeRes2014", "measurementData_Boehm_JProteomeRes2014.tsv"), DataFrame)
        validation_timepoints = unique(data.time)[2:5:end]
        if data_subset == "training"
            data = data[.∉(data.time, [validation_timepoints]), :]
        elseif data_subset == "validation"
            data = data[in.(data.time, [validation_timepoints]), :]
        end
    end

    data = unstack(data, :time, :observableId, :measurement)
    t = data[!, "time"]
    y_obs = identity.(transpose(Array(data[!, 2:end])))
    return t, y_obs, names(data[!,2:end])
end

function load_reference(problem_name)
    data = missing
    if occursin("ruoff", problem_name)
        data = read(joinpath("2_measurements", "Ruoff_synthetic_data", "reference_solution.csv"), DataFrame)
    end
    return data
end

using OrdinaryDiffEq: ODEProblem, Tsit5, KenCarp4, solve
using CSV, DataFrames

solver_name = ARGS[1]
if solver_name == "Tsit5"
    solver = Tsit5()
elseif solver_name == "KenCarp4"
    solver = KenCarp4()
end

function boehm_dynamics_full!(du, u, p, t)

    Epo_degradation_BaF3, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = p

    STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB = u

    BaF3_Epo = 1.25e-7 * exp(-Epo_degradation_BaF3 * t)
    v_nuc = 0.45
    v_cyt = 1.4

    # mechanistic terms
    # STAT5A
    du[1] =
        -2 * BaF3_Epo * STAT5A^2 * k_phos - BaF3_Epo * STAT5A * STAT5B * k_phos +
        2 * v_nuc / v_cyt * k_exp_homo * nucpApA +
        v_nuc / v_cyt * k_exp_hetero * nucpApB
    # STAT5B
    du[2] =
        -BaF3_Epo * STAT5A * STAT5B * k_phos - 2 * BaF3_Epo * STAT5B^2 * k_phos +
        v_nuc / v_cyt * k_exp_hetero * nucpApB +
        2 * v_nuc / v_cyt * k_exp_homo * nucpBpB
    # pApA
    du[3] = BaF3_Epo * STAT5A^2 * k_phos - k_imp_homo * pApA
    # pApB
    du[4] = BaF3_Epo * STAT5A * STAT5B * k_phos - k_imp_hetero * pApB
    # pBpB 
    du[5] = BaF3_Epo * STAT5B^2 * k_phos - k_imp_homo * pBpB
    # nucpApA
    du[6] = v_cyt / v_nuc * k_imp_homo * pApA - k_exp_homo * nucpApA
    # nucpApB
    du[7] = v_cyt / v_nuc * k_imp_hetero * pApB - k_exp_hetero * nucpApB
    # nucpBpB
    du[8] = v_cyt / v_nuc * k_imp_homo * pBpB - k_exp_homo * nucpBpB

end

initial_state = [143.867, 63.733, 0, 0, 0, 0, 0, 0]
p = [
    0.026982514033029,
    1.00067973851508E-05,
    0.006170228086381,
    0.0163679184468,
    97749.3794024716,
    15766.5070195731,
]
t = range(0, stop = 240, length = 100)
prob = ODEProblem{true}(boehm_dynamics_full!, initial_state, (t[1], t[end]), p)

# Measure the runtime and solve the problem
@time begin
    solution = solve(prob, solver, saveat = t, verbose = false)
end

solution_array = Array(solution)
# Prepare the data for CSV
df = DataFrame(
    time = solution.t,
    STAT5A = solution_array[1, :],
    STAT5B = solution_array[2, :],
    pApA = solution_array[3, :],
    pApB = solution_array[4, :],
    pBpB = solution_array[5, :],
    nucpApA = solution_array[6, :],
    nucpApB = solution_array[7, :],
    nucpBpB = solution_array[8, :],
)
# Save the solution to a CSV file
csv_filename = "solutions/boehm_solution_julia_$(solver_name).csv"
CSV.write(csv_filename, df)

# Store the runtime in a text file
runtime_file = "runtimes/runtime_boehm_julia_$(solver_name).txt"
open(runtime_file, "w") do file
    write(
        file,
        "Runtime: $(@elapsed solve(prob, solver, saveat = t, verbose = false)) seconds\n",
    )
end
println("Runtime saved to $runtime_file")

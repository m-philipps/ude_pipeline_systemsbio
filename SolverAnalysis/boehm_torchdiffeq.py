import torch
from torchdiffeq import odeint
import timeit
import sys
import numpy as np
import pandas as pd

solver = sys.argv[1]


# Define the Boehm equations
def boehm_dynamics(t, y):
    Epo_degradation_BaF3 = 0.026982514033029
    k_exp_hetero = 1.00067973851508e-05
    k_exp_homo = 0.006170228086381
    k_imp_hetero = 0.0163679184468
    k_imp_homo = 97749.3794024716
    k_phos = 15766.5070195731

    BaF3_Epo = 1.25e-07 * torch.exp(-Epo_degradation_BaF3 * t)

    STAT5A, STAT5B, pApB, pApA, pBpB, nucpApA, nucpApB, nucpBpB = y

    # static compartment volumes
    v_cyt = 1.4
    v_nuc = 0.45
    # differential equations
    dSTAT5Adt = (
        -2 * BaF3_Epo * STAT5A**2 * k_phos
        - BaF3_Epo * STAT5A * STAT5B * k_phos
        + 2 / v_cyt * (v_nuc * k_exp_homo * nucpApA)
        + v_nuc / v_cyt * k_exp_hetero * nucpApB
    )
    dSTAT5Bdt = (
        -BaF3_Epo * STAT5A * STAT5B * k_phos
        - 2 * BaF3_Epo * STAT5B**2 * k_phos
        + v_nuc / v_cyt * k_exp_hetero * nucpApB
        + 2 * v_nuc / v_cyt * k_exp_homo * nucpBpB
    )
    dpApA = BaF3_Epo * STAT5A**2 * k_phos - k_imp_homo * pApA
    dpApB = BaF3_Epo * STAT5A * STAT5B * k_phos - k_imp_hetero * pApB
    dpBpB = BaF3_Epo * STAT5B**2 * k_phos - k_imp_homo * pBpB
    dnucpApA = v_cyt / v_nuc * k_imp_homo * pApA - k_exp_homo * nucpApA
    dnucpApB = v_cyt / v_nuc * k_imp_hetero * pApB - k_exp_hetero * nucpApB
    dnucpBpB = v_cyt / v_nuc * k_imp_homo * pBpB - k_exp_homo * nucpBpB

    dydt = torch.tensor(
        [dSTAT5Adt, dSTAT5Bdt, dpApA, dpApB, dpBpB, dnucpApA, dnucpApB, dnucpBpB]
    )

    return dydt


# Set initial conditions
y0 = torch.tensor([143.867, 63.733, 0, 0, 0, 0, 0, 0])
t = torch.linspace(0, 240, 100)  # Time points to solve for

# Solve the ODE
t_start = timeit.default_timer()
solution = odeint(boehm_dynamics, y0, t, method=solver)
t_end = timeit.default_timer()

runtime = t_end - t_start
print(f"solver method {solver}, runtime in minutes: {runtime/60}")

# Convert solution to numpy array for easier handling with pandas
solution_np = solution.detach().numpy()

# Convert time points to numpy array and add as the first column
time_np = t.detach().numpy().reshape(-1, 1)
solution_with_time = np.hstack((time_np, solution_np))

# Create a DataFrame for easier CSV export
column_names = [
    "Time",
    "STAT5A",
    "STAT5B",
    "pApA",
    "pApB",
    "pBpB",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]
solution_df = pd.DataFrame(solution_with_time, columns=column_names)

# Save to CSV
csv_filename = f"solutions/boehm_solution_{solver}.csv"
solution_df.to_csv(csv_filename, index=False)

with open(f"runtimes/runtime_boehm_{solver}.txt", "w") as file:
    # Write the variable to the file
    file.write(f"{solver}\n")
    file.write(f"runtime in minutes\n")
    file.write(f"{runtime/60}\n")

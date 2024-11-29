import matplotlib.pyplot as plt
from petab.C import (
    ESTIMATE,
    NOMINAL_VALUE,
    PARAMETER_ID,
    PARAMETER_SCALE,
)


def mute_ruoff_v5_reaction(petab_problem):
    """Remove the reaction that is approximated by the UDE ANN."""
    for deleted_parameter_id in ["k_5"]:
        petab_problem.parameter_df.loc[deleted_parameter_id, ESTIMATE] = 0
        petab_problem.parameter_df.loc[deleted_parameter_id, NOMINAL_VALUE] = 0


def set_fix_values(x, pypesto_problem):
    """Set the fixed value for the parameters that are not to be estimated."""
    for i, val in zip(pypesto_problem.x_fixed_indices, pypesto_problem.x_fixed_vals):
        x[i] = val
    return x


def convert_to_nominal(df, petab_problem):
    parameter_df_free = petab_problem.parameter_df[
        petab_problem.parameter_df[ESTIMATE] == 1
    ].reset_index()
    parameter_scales = {
        p: s
        for p, s in zip(
            parameter_df_free[PARAMETER_ID], parameter_df_free[PARAMETER_SCALE]
        )
    }
    for p, s in parameter_scales.items():
        if s == "log10":
            df[p] = df[p].apply(lambda x: 10**x)
        elif s == "lin":
            return p
        else:
            raise KeyError("Parameter transformation is not implemented.")


def plot_function_values(fvals, fname):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(fvals)
    ax[1].plot(fvals)
    ax[1].set_yscale("log")
    ax[0].set_ylabel("objective function value [lin]")
    ax[1].set_ylabel("objective function value [log]")
    fig.suptitle("Objective function values")
    fig.savefig(fname)

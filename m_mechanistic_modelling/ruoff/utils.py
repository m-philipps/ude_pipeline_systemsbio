from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

import amici

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir / "1_mechanistic_model"))
from reference_ruoff import TRAINING_ENDPOINT, TEST_ENDPOINT, OBSERVED


def rdata_to_df(
    rdata: amici.numpy.ReturnDataView,
    amici_model,
    time: list,
    observables: bool = False,
):
    df = pd.DataFrame({"time": time})
    for i, species_id in enumerate(amici_model.getStateIds()):
        df[species_id] = rdata["x"][:, i]
    if observables:
        for i, obs_id in enumerate(amici_model.getObservableIds()):
            df[obs_id] = rdata["y"][:, i]
    return df


def unscale(p, scale):
    if scale == "log10":
        return 10**p
    elif scale == "log":
        return np.exp(p)
    elif scale == "lin":
        return p
    else:
        raise NotImplementedError(f'Scale "{scale}" is not implemented yet.')


def unscale_vector(p, pypesto_problem) -> list:
    "Return a vector of optimised parameter values on linear scale."
    scales = pypesto_problem.x_scales
    return [unscale(p, scale) for p, scale in zip(p, scales)]


def simulate_optimised_model(
    pypesto_problem, p_opt, unscale_parameters: bool = True, observables: bool = False
):
    if unscale_parameters:
        # get opt'd parameters on linear scale
        p_opt = unscale_vector(p_opt, pypesto_problem)
    amici_model = pypesto_problem.objective.amici_model
    # set narrow timepoints
    t_step = 0.01
    t_ref = np.around(np.arange(0, TEST_ENDPOINT + t_step, t_step), decimals=2)
    amici_model.setTimepoints(t_ref)
    # set optimised parameters
    for xname, p_ in zip(pypesto_problem.x_names, p_opt):
        xname = xname.replace("sd", "noiseParameter1")
        amici_model.setParameterByName(xname, p_)
    # simulate
    solver = amici_model.getSolver()
    rdata = amici.runAmiciSimulation(amici_model, solver)
    df = rdata_to_df(
        rdata=rdata,
        amici_model=amici_model,
        time=t_ref,
        observables=observables,
    )
    return df


def get_reference():
    fp = base_dir / "2_measurements" / "Ruoff_synthetic_data" / "reference_solution.csv"
    return pd.read_csv(fp)


# == evaluation ==

REFERENCE = get_reference()


def NMSE(prediction, reference):
    return ((prediction - reference) ** 2).mean() / reference.mean()


def NMSE_observables(p_opt, pypesto_problem, unscale_parameters: bool = True):
    # simulate
    sim = simulate_optimised_model(
        pypesto_problem, p_opt, unscale_parameters=unscale_parameters
    )
    # subset to test regime and observables
    pred = sim.query("@TRAINING_ENDPOINT < time")[OBSERVED]
    ref = REFERENCE.query("@TRAINING_ENDPOINT < time")[OBSERVED]
    # mean(abs2, (pred.-obs)./μ)
    nmse = ((pred - ref) ** 2).div(ref.mean(axis=0)).values.mean()
    return nmse


def extract_from_result(result, key):
    try:
        return result[key]
    except KeyError:
        return None


def plot_fval_nmse(result, noise, n_dp):
    """Plot fvals (NLL on training data) and NMSE (test data)."""
    # get fvals and nmse
    fvals = result.optimize_result.fval
    fval_min = min(fvals)
    fvals = [f - fval_min + 1 for f in fvals]
    nmses = [extract_from_result(r, "NMSE_test") for r in result.optimize_result]
    # remove trailing nones for NMSEs
    while nmses[-1] is None:
        del nmses[-1]
    # name a list of all nmses that are nan to plot the corresp. fval
    nmses_nans = []
    for nmse, fval in zip(nmses, fvals):
        if np.isnan(nmse):
            nmses_nans.append(fval)
        else:
            nmses_nans.append(np.nan)
    # calculate the rank correlation between fval, nmse
    corr = spearmanr(fvals[: len(nmses)], nmses, axis=0, nan_policy="omit")
    # plot
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3))
    fig.suptitle(
        "{}% noise, {} data points: Correlation {} (p={})".format(
            noise, n_dp, round(corr.correlation, 2), round(corr.pvalue, 2)
        ),
        fontsize="medium",
    )
    # best 20 starts
    ax[0].set_title("Best 20 starts")
    ax[0].plot(fvals[:20], label="neg log likelihood (rel. to best)")
    ax[0].plot(nmses[:20], ".", label="NMSE")
    # indicate NAN NMSEs
    ax[0].plot(nmses_nans[:20], "or", markerfacecolor="none", label="NMSE = NAN")
    ax[0].set_yscale("log")
    # all converged starts
    ax[1].set_title("All starts")
    ax[1].plot(fvals, label="neg log likelihood (rel. to best)")
    ax[1].plot(nmses, ".", label="NMSE")
    ax[1].plot(nmses_nans[:20], "or", markerfacecolor="none", label="NMSE = NAN")
    ax[1].set_yscale("log")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_fit(
    petab_problem,
    pypesto_problem,
    p_opt,
    noise,
    n_datapoints,
    plot_noise: bool,
    unscale_parameters: bool = True,
):
    df = simulate_optimised_model(
        pypesto_problem, p_opt, unscale_parameters=unscale_parameters
    )[["time"] + OBSERVED]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    fig.suptitle(f"Best Fit ({noise}% noise, {n_datapoints} data points)")

    for obs, axis in zip(OBSERVED, ax):
        axis.set_title(f"${obs}$", y=0.8)
        # prediction
        axis.plot(df["time"], df[obs], label="Simulation")
        # training data
        obs_petab_id = f"{obs}_obs"
        dfm = petab_problem.measurement_df.query("observableId == @obs_petab_id")
        axis.plot(
            dfm["time"],
            dfm["measurement"],
            "o",
            color="dimgrey",
            markersize="2",
            label="Data",
        )
        # reference
        axis.plot(
            REFERENCE["time"],
            REFERENCE[obs],
            linestyle="dashed",
            color="dimgrey",
            label="Reference",
        )
        if plot_noise:
            # get the opt. noise parameters
            if unscale_parameters:
                p_opt = unscale_vector(p_opt, pypesto_problem)
            p_opt_dict = dict(zip(pypesto_problem.x_names, p_opt))
            # plot
            y = df[obs]
            std = p_opt_dict["sd_" + obs_petab_id]
            axis.fill_between(
                df["time"],
                (y - std),
                (y + std),
                color="tab:blue",
                alpha=0.4,
                edgecolor=None,
                label=f"+/- std {round(std, 4)}",
            )
    ax[-1].set_xlabel("Time")
    ax[-1].legend(loc=4)

    fig.tight_layout()
    return fig, ax

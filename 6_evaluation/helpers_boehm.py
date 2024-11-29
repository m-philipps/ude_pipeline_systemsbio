import sys
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from petab.C import TIME, MEASUREMENT

base_dir = Path(__file__).resolve().parents[1]
dir_1 = base_dir / "1_mechanistic_model"
dir_data = base_dir / "2_measurements"
sys.path.append(str(dir_1))

from reference_boehm import (
    model_name_petab,
    SPECIES_IDS,
    OBSERVABLES_IDS,
    NOISE_PARAMETER_IDS,
    T_START,
    T_END,
)


def update_summary_with_trainval_metrics(summary):
    "Calculate the trainval metrics for the given experiment summary."
    for metric in ["negLL_obs", "mse_obs", "nmse_obs"]:
        summary[f"{metric}_trainval"] = (
            summary[f"{metric}_train"] + summary[f"{metric}_val"]
        ) / 2
    return summary


def load_exp_summary(dir_exp_output: Path):
    summary = pd.read_csv(dir_exp_output / "experiment_summary.csv")
    # drop the runs that did not finish
    summary = summary.dropna(subset=NOISE_PARAMETER_IDS)
    return summary


def inverse_transform(x, lb=1e-5, ub=1e5):
    """Inverse parameter transform for the tanh bounds."""
    lb = lb * (1-1e-6)
    ub = ub * (1+1e-6)
    return lb + (np.tanh(x-1.73) + 1)/2*(ub-lb)


## plotting trajectories


def load_measurements() -> pd.DataFrame:
    fp = dir_1 / model_name_petab / "measurementData_Boehm_JProteomeRes2014.tsv"
    return pd.read_csv(fp, delimiter="\t")


def load_simulation(experiment_output_path, model_id):
    fp = experiment_output_path / f"ude_{model_id}" / "simulation.csv"
    sim = None
    try:
        sim = pd.read_csv(fp)
    except FileNotFoundError:
        print(fp, " not available.")
    return sim


def plot_obs_fits_together(
    simulation: pd.DataFrame,
    sd_parameters: Optional[dict] = None,
) -> tuple[plt.Figure, plt.Axes.axes]:
    """
    Plot best fit and measurements.

    Parameters
    ----------
    simulation:
        The simulation dataframe, expected columns are 'Time' and the observables ids.
    show_noise:
        If supplied must be a dictionary mapping the observable id to the
        standard deviation used for plotting the noise.
    """
    # measurement data
    df_data = load_measurements()
    # plot
    fig, ax = plt.subplots()
    for observable_id, colour in zip(
        OBSERVABLES_IDS, ("dodgerblue", "tab:orange", "mediumseagreen")
    ):
        # measurements
        ax.plot(
            df_data.query(f"observableId == @observable_id")[TIME],
            df_data.query(f"observableId == @observable_id")[MEASUREMENT],
            label=f"Data {observable_id[:-4]}",
            marker="o",
            linestyle="",
            markersize=4,
            color=colour,
        )
        # fit
        ax.plot(
            simulation["Time"],
            simulation[observable_id],
            label=f"Fit {observable_id[:-4]}",
            color=colour,
        )
        # noise
        if sd_parameters:
            y = simulation[observable_id]
            sd = sd_parameters["sd_" + observable_id]
            ax.fill_between(
                simulation["Time"],
                (y - sd),
                (y + sd),
                color=colour,
                alpha=0.2,
                edgecolor=None,
                # label=f"+/- std {round(sd, 3)}",
            )
    ax.set_ylabel(
        "Observables",
        # rotation="horizontal",
        fontsize=12,
        labelpad=15,
    )

    ax.set_xlabel("Time")
    ax.legend(prop={"size": 6})
    return fig, ax


def plot_obs_fits_individually(
    simulation: pd.DataFrame,
    sd_parameters: Optional[dict] = None,
) -> tuple[plt.Figure, plt.Axes.axes]:
    """
    Plot best fit and measurements.

    Parameters
    ----------
    simulation:
        The simulation dataframe, expected columns are 'Time' and the observables ids.
    show_noise:
        If supplied must be a dictionary mapping the observable id to the
        standard deviation used for plotting the noise.
    """
    # measurement data
    df_data = load_measurements()
    # plot
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
    for observable_id, ax, colour in zip(
        OBSERVABLES_IDS, axes, ("dodgerblue", "tab:orange", "mediumseagreen")
    ):
        # measurements
        ax.plot(
            df_data.query(f"observableId == @observable_id")[TIME],
            df_data.query(f"observableId == @observable_id")[MEASUREMENT],
            label="Data",
            marker="o",
            linestyle="",
            markersize=4,
            color=colour,
        )
        # fit
        ax.plot(
            simulation["Time"],
            simulation[observable_id],
            label="Fit",
            color=colour,
        )
        # noise
        if sd_parameters:
            y = simulation[observable_id]
            sd = sd_parameters["sd_" + observable_id]
            ax.fill_between(
                simulation["Time"],
                (y - sd),
                (y + sd),
                color=colour,
                alpha=0.2,
                edgecolor=None,
                label=f"+/- std {round(sd, 3)}",
            )
        ax.set_title(observable_id)
        ax.set_xlabel("Time")
        ax.legend(prop={"size": 6})
    return fig, ax


def plot_state_space(
    simulation: pd.DataFrame, species_ids: list[str]
) -> tuple[plt.Figure, plt.Axes.axes]:
    """
    Plot the full simulation for all species.

    Parameters
    ----------
    simulation:
        The simulation dataframe, expected columns are 'Time' and the species ids.
    species_ids:
        The ids of the species to be plotted.
    """
    # plot
    fig, axes = plt.subplots(
        nrows=math.ceil(len(species_ids) / 2), ncols=2, sharex=True
    )
    for ax, sid in zip(axes.flatten(), species_ids):
        # plot simulation
        ax.plot(
            simulation["Time"],
            simulation[sid],
            label="Simulation",
            color="tab:blue",
        )
        ax.set_title(sid,  pad=-50, fontsize="small")
    for ax in axes[-1]:
        ax.set_xlabel("Time")
    axes[0][0].legend()
    return fig, axes

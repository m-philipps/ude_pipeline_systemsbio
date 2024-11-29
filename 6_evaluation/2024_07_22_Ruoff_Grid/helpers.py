import sys
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd

base_dir = Path(__file__).resolve().parents[2]
dir_1 = base_dir / "1_mechanistic_model"
dir_data = base_dir / "2_measurements"
sys.path.append(str(dir_1))

from reference_ruoff import (
    SPECIES_IDS,
    OBSERVED,
    OBSERVABLES_IDS,
    NOISE_PARAMETER_IDS,
    TRAINING_ENDPOINT,
    TEST_ENDPOINT,
)


def load_measurements(noise, n_datapoints) -> tuple[pd.DataFrame]:
    fp_training = (
        dir_data
        / "Ruoff_synthetic_data"
        / f"data_{n_datapoints}_noise_{noise}_training.tsv"
    )
    fp_validation = (
        dir_data
        / "Ruoff_synthetic_data"
        / f"data_{n_datapoints}_noise_{noise}_validation.tsv"
    )
    df_training = pd.read_csv(fp_training, delimiter="\t")
    df_validation = pd.read_csv(fp_validation, delimiter="\t")
    return df_training, df_validation


def load_reference() -> pd.DataFrame:
    fp_reference = dir_data / "Ruoff_synthetic_data" / "reference_solution.csv"
    return pd.read_csv(fp_reference)


def load_simulation(experiment_output_path, model_id):
    fp = experiment_output_path / "result" / f"ude_{model_id}" / "simulation.csv"
    sim = None
    try:
        sim = pd.read_csv(fp)
    except FileNotFoundError:
        print(fp, " not available.")
    return sim


def load_exp_summary(dir_exp_output: Path):
    summary = pd.read_csv(dir_exp_output / "experiment_summary.csv")
    # drop the runs that did not finish
    summary = summary.dropna(subset=NOISE_PARAMETER_IDS)
    return summary


def plot_fit(
    simulation: pd.DataFrame,
    noise,
    n_datapoints,
    predict: bool,
    sd_parameters: Optional[dict] = None,
) -> tuple[plt.Figure, plt.Axes.axes]:
    """
    Plot best fit and measurements.

    Parameters
    ----------
    simulation:
        The simulation dataframe, expected columns are 'Time' and the species ids.
    noise:
        The noise in the measurements.
    n_datapoints:
        The number of measurements that per observable in the dataset.
    predict:
        If true, shows the model simulation beyond the training data.
    show_noise:
        If supplied must be a dictionary mapping the observable id to the
        standard deviation used for plotting the noise.
    """
    # measurement data
    df_data_training, df_data_validation = load_measurements(noise, n_datapoints)
    if predict:
        t_end = TEST_ENDPOINT
        ref = load_reference().query("time > @TRAINING_ENDPOINT")
    else:
        t_end = TRAINING_ENDPOINT
    # plot
    fig, axes = plt.subplots(nrows=2, sharex=True)
    for ax, observable_id, petab_obs_id in zip(axes, OBSERVED, OBSERVABLES_IDS):
        for label, df, color in zip(
            ["training", "validation"],
            [df_data_training, df_data_validation],
            ["mediumblue", "skyblue"],
        ):
            # measurements
            ax.plot(
                df.query(f"observableId == @petab_obs_id")["time"],
                df.query(f"observableId == @petab_obs_id")["measurement"],
                label=f"{label} data",
                marker="o",
                linestyle="",
                markersize=4,
                color=color,
            )
        # fit
        sim_train_regime = simulation[simulation["Time"] <= t_end]
        ax.plot(
            sim_train_regime["Time"],
            sim_train_regime[observable_id],
            label="Simulation",
            color="k",
        )
        ax.set_ylabel(
            f"${observable_id}$",
            rotation="horizontal",
            fontsize=12,
            labelpad=15,
        )
        # noise
        if sd_parameters:
            y = sim_train_regime[observable_id]
            sd = sd_parameters["sd_" + petab_obs_id]
            ax.fill_between(
                sim_train_regime["Time"],
                (y - sd),
                (y + sd),
                color="grey",
                alpha=0.2,
                edgecolor=None,
                label=f"+/- std {round(sd, 3)}",
            )
        # true solution for prediction
        if predict:
            ax.plot(
                ref["time"],
                ref[observable_id],
                label="True solution",
                linestyle="dashed",
                color="dimgrey",
            )

    axes[1].set_xlabel("Time")
    axes[0].legend(prop={'size': 6})
    if sd_parameters:
        # show estimated noise for both observables
        axes[1].legend(prop={'size': 6})
    return fig, axes


def plot_full_state_space(
    simulation: pd.DataFrame,
    predict: bool,
) -> tuple[plt.Figure, plt.Axes.axes]:
    """
    Plot best fit and measurements.

    Parameters
    ----------
    simulation:
        The simulation dataframe, expected columns are 'Time' and the species ids.
    noise:
        The noise in the measurements.
    n_datapoints:
        The number of measurements that per observable in the dataset.
    predict:
        If true, shows the model simulation beyond the training data.
    """
    ref = load_reference()
    if not predict:
        ref = ref.query("time <= @TRAINING_ENDPOINT")
        simulation = simulation[simulation["Time"] <= TRAINING_ENDPOINT]
    # plot
    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True)
    for ax, species_id in zip(axes.flatten(), SPECIES_IDS):
        # plot simulation
        ax.plot(
            simulation["Time"],
            simulation[species_id],
            label="Simulation",
            color="tab:blue",
        )
        ax.set_ylabel(
            species_id,
            rotation="horizontal",
            fontsize=12,
            labelpad=15,
        )
        # true solution
        ax.plot(
            ref["time"],
            ref[species_id],
            label="True solution",
            linestyle="dashed",
            color="dimgrey",
        )
    for ax in axes[-1]:
        ax.set_xlabel("Time")
    axes[0][0].legend()
    return fig, axes

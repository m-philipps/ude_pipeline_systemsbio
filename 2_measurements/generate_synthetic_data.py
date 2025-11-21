"""Simulate data for the synthetic Ruoff problem."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
import sys
import petab
import petab.C as C
import amici
import amici.petab_simulate
from amici.petab_import import import_petab_problem

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "1_mechanistic_model"))
from reference_ruoff import (
    NOISE_PARAMETER_IDS,
    SPECIES_IDS,
    OBSERVABLES_IDS,
    OBSERVABLES_NAMES,
    DATASET_SIZES,
    NOISE_PERCENTAGES,
    TRAINING_ENDPOINT,
    TEST_ENDPOINT,
)

petab_dir = base_dir / "1_mechanistic_model" / "Ruoff_BPC2003"
data_dir = Path(__file__).parent / "Ruoff_synthetic_data"
data_dir.mkdir(exist_ok=True)

rng = np.random.default_rng(seed=2)


def get_blank_measurement_df(t, t_n):
    "Set up an empty petab measurement df."
    observable_ids = []
    [observable_ids.extend([oi] * t_n) for oi in OBSERVABLES_IDS]
    noise_parameter_ids = []
    [noise_parameter_ids.extend([ni] * t_n) for ni in NOISE_PARAMETER_IDS]
    return pd.DataFrame(
        {
            C.OBSERVABLE_ID: observable_ids,
            C.SIMULATION_CONDITION_ID: ["model1_data1"] * (t_n * 2),
            C.MEASUREMENT: [np.nan] * (t_n * 2),  # to be simulated
            C.TIME: np.hstack([t, t]),
            C.NOISE_PARAMETERS: noise_parameter_ids,
            C.DATASET_ID: ["data1"] * (t_n * 2),
        }
    )


def plot_sampled_training_data(
    df,
    line: bool = False,
    legend: bool = True,
    reference=None,
    return_axes: bool = False,
):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 3))
    for obs_id, obs_name, axis in zip(OBSERVABLES_IDS, OBSERVABLES_NAMES, ax):
        index = df.observableId == obs_id
        axis.scatter(
            df[C.TIME].loc[index],
            df[C.MEASUREMENT].loc[index],
            s=4,
            label="Sampled",
            c="mediumblue",
        )
        if line:
            axis.plot(
                df[C.TIME].loc[index], df[C.MEASUREMENT].loc[index], c="mediumblue"
            )
        if reference is not None:
            index = reference.observableId == obs_id
            axis.plot(
                reference[C.TIME].loc[index],
                reference[C.MEASUREMENT].loc[index],
                c="silver",
                linestyle="--",
                label="Reference",
            )
        axis.set_ylabel(obs_name)
        plt.close()
    ax[1].set_xlabel("Time")
    if legend:
        ax[1].legend()
    if return_axes:
        return fig, ax
    return fig


def plot_sampled_data(df_train, df_val, reference, return_axes: bool = False) -> None:
    """
    Plot sampled training/validation data and the reference.
    """
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    for obs_id, obs_name, ax in zip(
        OBSERVABLES_IDS,
        OBSERVABLES_NAMES,
        axes,
    ):
        for label, df, color in zip(
            ["Training", "Validation"],
            [df_train, df_val],
            ["mediumblue", "mediumpurple"],
        ):
            # subset to new training regime
            ax.plot(
                df.query(f"observableId == @obs_id")[C.TIME],
                df.query(f"observableId == @obs_id")[C.MEASUREMENT],
                label=f"{label} data",
                marker="o",
                linestyle="",
                markersize=4,
                color=color,
            )
        # reference
        index = reference.observableId == obs_id
        # subset to training regime
        reference_ = reference.query(f"{C.TIME} < @TRAINING_ENDPOINT")
        ax.plot(
            reference_[C.TIME].loc[index],
            reference_[C.MEASUREMENT].loc[index],
            c="silver",
            linestyle="--",
            label="Reference",
        )
        ax.set_ylabel(obs_name, fontsize=12)
    axes[1].set_xlabel("Time", fontsize=12)
    axes[0].legend(fontsize=12, loc="right")
    if return_axes:
        return fig, ax
    return fig


if __name__ == "__main__":
    # reference
    petab_problem = petab.Problem.from_yaml(petab_dir / "Ruoff_BPC2003_noise_5.yaml")
    amici_model = import_petab_problem(petab_problem)
    # set narrow timepoints
    t_step = 0.01
    t_ref = np.around(np.arange(0, TEST_ENDPOINT + t_step, t_step), decimals=2)
    amici_model.setTimepoints(t_ref)
    # simulate
    solver = amici_model.getSolver()
    rdata = amici.runAmiciSimulation(amici_model, solver)
    ref_df = pd.DataFrame({C.TIME: t_ref})
    for i, species_id in enumerate(SPECIES_IDS):
        ref_df[species_id] = rdata["x"][:, i]
    ref_df.to_csv(data_dir / "reference_solution.csv", index=False, header=True)

    # training and validation data
    for n_datapoints in DATASET_SIZES:
        # vector of time points
        t_step = TEST_ENDPOINT / (n_datapoints)
        t = np.around(np.arange(0, TEST_ENDPOINT + t_step, t_step), decimals=2)
        tn = len(t)
        t_train_val = [ti for ti in t if ti <= TRAINING_ENDPOINT]
        if n_datapoints == 25:
            print("8 datapoints/observable -> train/val split 5:3")
            t_training = [t_train_val[ti] for ti in [0, 2, 3, 5, 7]]
            t_validation = [t_train_val[ti] for ti in [1, 4, 6]]
        else:  
            # 1:1 split training:validation
            t_training = t_train_val[::2]
            t_validation = t_train_val[1::2]
        assert not set(t_training) & set(
            t_validation
        ), "Overlap in training and validation set"

        # simulate
        measurement_df_no_noise = get_blank_measurement_df(t, tn)
        # fill measurements from reference
        measurements = deepcopy(
            ref_df[ref_df.time.isin(t)]["N_2"].tolist()
            + ref_df[ref_df.time.isin(t)]["A_3"].tolist()
        )
        measurement_df_no_noise[C.MEASUREMENT] = measurements

        # add noise
        obs_means = measurement_df_no_noise.groupby(
            by="observableId"
        ).measurement.mean()
        obs_means = obs_means.loc[(["N_2_obs"] * tn + ["A_3_obs"] * tn)]
        for noise_percent in NOISE_PERCENTAGES:
            noise_level = noise_percent / 100
            noisy_df = deepcopy(measurement_df_no_noise)
            if noise_level:
                noisy_df[C.MEASUREMENT] = rng.normal(
                    loc=noisy_df[C.MEASUREMENT],
                    scale=obs_means * noise_level,
                )
            # set negative measurement points to 0
            noisy_df.loc[noisy_df[C.MEASUREMENT] < 0, C.MEASUREMENT] = 1e-10

            # validation
            fh_validation = (
                data_dir / f"data_{n_datapoints}_noise_{noise_percent}_validation.tsv"
            )
            # every second entry (starting at 2nd entry)
            noisy_df_validation = noisy_df[noisy_df[C.TIME].isin(t_validation)]
            noisy_df_validation.to_csv(fh_validation, sep="\t", index=False)

            # training
            fh_training = (
                data_dir / f"data_{n_datapoints}_noise_{noise_percent}_training.tsv"
            )
            noisy_df_training = noisy_df[noisy_df[C.TIME].isin(t_training)]
            noisy_df_training.to_csv(fh_training, sep="\t", index=False)

            # visualise
            fig = plot_sampled_data(
                noisy_df_training,
                noisy_df_validation,
                reference=measurement_df_no_noise,
            )
            fig.suptitle(f"Ruoff sampled data, {noise_percent}% noise", fontsize=15)
            fig.tight_layout()
            fig.savefig(
                str(data_dir / f"data_{n_datapoints}_noise_{noise_percent}.svg")
            )

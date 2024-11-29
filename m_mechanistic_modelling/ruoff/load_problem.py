import sys
from pathlib import Path

import petab
import pypesto
import pypesto.petab
import numpy as np
import pandas as pd

dir_pipeline = Path(__file__).resolve().parents[2]
dir_1 = dir_pipeline / "1_mechanistic_model"
dir_2 = dir_pipeline / "2_measurements"
dir_3 = dir_pipeline / "3_start_points"
dir_m = dir_pipeline / "m_mechanistic_modelling"
for d in [dir_pipeline, dir_1, dir_2, dir_3]:
    sys.path.append(str(d))

from reference_ruoff import (
    NOISE_PERCENTAGES,
    DATASET_SIZES,
    model_name_petab,
    model_name,
)


def load_synthetic_data(noise: int, n_datapoints: int) -> pd.DataFrame:
    """Load training and validation data to be used for MM training."""
    dirpath = dir_2 / "Ruoff_synthetic_data"
    df_train = pd.read_csv(
        dirpath / f"data_{n_datapoints}_noise_{noise}_training.tsv",
        sep="\t",
    )
    df_val = pd.read_csv(
        dirpath / f"data_{n_datapoints}_noise_{noise}_validation.tsv",
        sep="\t",
    )
    return pd.concat([df_train, df_val]).reset_index(drop=True)


def get_startpoints(noise):
    return np.loadtxt(
        dir_3 / (model_name + "_full") / f"startpoints_lhs_noise_{noise}.csv",
        delimiter="\t",
    )


petab_problems = {n: {} for n in NOISE_PERCENTAGES}
pypesto_problems = {n: {} for n in NOISE_PERCENTAGES}

for noise in NOISE_PERCENTAGES:
    fp_petab_yaml = dir_1 / model_name_petab / f"{model_name_petab}_noise_{noise}.yaml"

    for dps in DATASET_SIZES:
        petab_problem = petab.Problem.from_yaml(fp_petab_yaml)
        # get synthetic data
        data = load_synthetic_data(noise, n_datapoints=dps)
        petab_problem.measurement_df = data
        # get pypesto problem
        amici_model_name = f"{model_name_petab}_{noise}_{dps}"
        amici_model_dir = dir_m / "amici_models" / amici_model_name
        pypesto_importer = pypesto.petab.PetabImporter(
            petab_problem,
            output_folder=amici_model_dir,
            model_name=amici_model_name,
        )
        pypesto_problem = pypesto_importer.create_problem()
        # set shared start points
        pypesto_problem.set_x_guesses(
            pypesto_problem.get_full_vector(get_startpoints(noise))
        )
        # stash
        petab_problems[noise][dps] = petab_problem
        pypesto_problems[noise][dps] = pypesto_problem

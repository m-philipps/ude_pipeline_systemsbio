"""Sample LHS start points for the Ruoff model."""

import os
import sys
from pathlib import Path
import json

import numpy as np
import numpy.random as random
import pandas as pd

import pypesto
import pypesto.petab
import petab

# add required pipeline directories
base_dir = Path(__file__).resolve().parents[1]
dir_1 = base_dir / "1_mechanistic_model"
dir_3 = Path(__file__).resolve().parent
for dirpath in (base_dir, dir_1):
    sys.path.append(str(dirpath))
# import from pipeline
import utils
from reference_ruoff import (
    model_name,
    model_name_petab,
    PARAMETERS_IDS,
    NOISE_PERCENTAGES,
    NOISE_PARAMETER_IDS,
)

# load references
with open(dir_3 / "reference_ruoff.json", "r") as jf:
    hp_options = json.load(jf)
with open(base_dir / "problems.json", "r") as jf:
    problems = json.load(jf)
ude_id = "ruoff_atp_consumption"

random.seed(2)
n_start_points = hp_options["n_starts"]

# import petab problem
fp_petab_yaml = dir_1 / model_name_petab / (model_name_petab + "_noise_5.yaml")
petab_problem = petab.Problem.from_yaml(fp_petab_yaml)
# get pypesto problem
amici_model_dir = dir_3 / "amici_models" / model_name
pypesto_importer = pypesto.petab.PetabImporter(
    petab_problem,
    output_folder=amici_model_dir,
    model_name=model_name,
)
pypesto_problem = pypesto_importer.create_problem()

# sample start points
startpoint_method = pypesto.startpoint.LatinHypercubeStartpoints(
    use_guesses=False,
    check_fval=True,
    check_grad=True,
)
startpoints = startpoint_method(
    n_starts=n_start_points,
    problem=pypesto_problem,
)

for noise, factor in zip(NOISE_PERCENTAGES, (1, 2, 4, 8)):
    parameters = np.array(pypesto_problem.x_names)[pypesto_problem.x_free_indices]
    df = pd.DataFrame(startpoints, columns=parameters)

    # scale the standard deviation parameter
    for noise_parameter in NOISE_PARAMETER_IDS:
        print(noise_parameter, df[noise_parameter].mean(), df[noise_parameter].max())
        df[noise_parameter] = df[noise_parameter] * factor
        print("  after scaling:", df[noise_parameter].mean(), df[noise_parameter].max())

    # save for mechanistic model
    # output_dir = dir_3 / (model_name + "_full")
    # os.makedirs(output_dir, exist_ok=True)
    # np.savetxt(
    #     output_dir / f"startpoints_lhs_noise_{noise}.csv", df.values, delimiter="\t"
    # )

    # get UDE's mechanistic parameters
    ude_p_m_indicators = problems[ude_id]["mechanistic_parameters"]
    ude_parameter_ids = np.array(PARAMETERS_IDS)[
        np.array(ude_p_m_indicators, dtype=bool)
    ]
    # drop muted parameter
    df = df[ude_parameter_ids]

    # convert to nominal
    utils.mute_ruoff_v5_reaction(petab_problem)
    utils.convert_to_nominal(df, petab_problem)

    # save for UDE
    output_dir = dir_3 / ude_id
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / f"startpoints_lhs_noise_{noise}.csv", index=False)

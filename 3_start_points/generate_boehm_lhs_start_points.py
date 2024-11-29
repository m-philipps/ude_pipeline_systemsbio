"""Sample LHS start points for the Boehm model."""

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

boehm_problem = "boehm_observable_ab_ratio"

# add required pipeline directories
base_dir = Path(__file__).resolve().parents[1]
dir_1 = base_dir / "1_mechanistic_model"
dir_3 = Path(__file__).resolve().parent
for dirpath in (base_dir, dir_1):
    sys.path.append(str(dirpath))
# import from pipeline
import utils
import reference_boehm as ref

# load references
with open(dir_3 / f"reference_{boehm_problem}.json", "r") as jf:
    hp_options = json.load(jf)
with open(base_dir / "problems.json", "r") as jf:
    problems = json.load(jf)
boehm_ids = problems["boehm_ids"]

random.seed(2)
n_start_points = hp_options["n_starts"]

# generate start points for all mechanistic parameters
model_petab_name = "Boehm_JProteomeRes2014"
fp_petab_yaml = dir_1 / ref.model_name_petab / (model_petab_name + ".yaml")
petab_problem = petab.Problem.from_yaml(fp_petab_yaml)
# get pypesto problem
amici_model_dir = dir_3 / "amici_models" / model_petab_name
pypesto_importer = pypesto.petab.PetabImporter(
    petab_problem,
    output_folder=amici_model_dir,
    model_name=model_petab_name,
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
parameters = np.array(pypesto_problem.x_names)[pypesto_problem.x_free_indices]
df = pd.DataFrame(startpoints, columns=parameters)

# convert to nominal
utils.convert_to_nominal(df, petab_problem)

# save
output_dir = dir_3 / boehm_problem  # "boehm_full"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_dir / "startpoints_lhs.csv", index=False)

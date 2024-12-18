"""Sample hyperparameters."""

import sys
from pathlib import Path
import json
import numpy as np
import numpy.random as random
import pandas as pd
import pypesto
from pypesto.startpoint.latin_hypercube import latin_hypercube

# add required pipeline directories
base_dir = Path(__file__).resolve().parents[1]
dir_1 = base_dir / "1_mechanistic_model"
dir_3 = Path(__file__).resolve().parent
for dirpath in (base_dir, dir_1):
    sys.path.append(str(dirpath))

# load references
with open(dir_3 / "reference_ruoff.json", "r") as jf:
    hp_options = json.load(jf)
ude_id = "ruoff_atp_consumption"

random.seed(2)
n_samples = hp_options["n_starts"]
hp_options = hp_options["hp_options"]

# sample FNN depth
fnn_depths = (
    latin_hypercube(
        n_samples,
        lb=np.array(hp_options["fnn_depth"][0]),
        ub=np.array(hp_options["fnn_depth"][-1] + 1),
    )
    .astype(int)
    .reshape(1, -1)[0]
)
assert 5 > max(fnn_depths)

# sample FNN width
n_width_options = len(hp_options["fnn_width"])
fnn_widths = (
    latin_hypercube(n_samples, lb=np.array(0), ub=np.array(n_width_options))
    .astype(int)
    .reshape(1, -1)[0]
)
assert n_width_options > max(fnn_widths)

# sample activation functions
n_act_options = len(hp_options["activation"])
activations = (
    latin_hypercube(n_samples, lb=np.array(0), ub=np.array(n_act_options))
    .astype(int)
    .reshape(1, -1)[0]
)
assert n_act_options > max(activations)

# sample regularisation strengths on log scale
regularisation_strengths = latin_hypercube(
    n_samples, lb=np.array(np.log(1e-3)), ub=np.array(np.log(1e1))
).reshape(1, -1)[0]
regularisation_strengths = np.exp(regularisation_strengths)

# sample learning rates on log scale
lrs_adam = latin_hypercube(
    n_samples,
    lb=np.array(np.log(hp_options["lr_adam"][0])),
    ub=np.array(np.log(hp_options["lr_adam"][-1])),
).reshape(1, -1)[0]
lrs_adam = np.exp(lrs_adam)

# sample normalisation
normalisations = (
    latin_hypercube(n_samples, lb=np.array(0), ub=np.array(2))
    .astype(int)
    .reshape(1, -1)[0]
)
assert 2 > max(normalisations)

# convert to dataframe matchin Nina's format
df = pd.DataFrame(
    {
        "hidden_layers": fnn_depths,
        "hidden_neurons": fnn_widths,
        "act_fct": activations,
        "nn_input_normalization": normalisations,
        "λ_reg": regularisation_strengths,
        "lr_adam": lrs_adam,
    }
)

# map to options
df["hidden_neurons"] = df["hidden_neurons"].map(
    dict(enumerate(hp_options["fnn_width"]))
)
df["act_fct"] = df["act_fct"].map(dict(enumerate(hp_options["activation"])))

# randomly replace some regularisation strenght with 0 to disable reg.
fraction_without_reg = 0.2
factors = random.choice(
    [0, 1], size=n_samples, p=[fraction_without_reg, 1 - fraction_without_reg]
)
df["λ_reg"] = df["λ_reg"] * factors

output_dir = dir_3 / ude_id
df.to_csv(output_dir / "hps.csv", index=False)

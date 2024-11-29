"Plot fits by list of ude nrs."

import numpy as np
from pathlib import Path
from helpers_ruoff import *

dir_pipeline = Path(__file__).resolve().parents[2]
dir_1 = dir_pipeline / "1_mechanistic_model"
sys.path.append(str(dir_1))
from reference_ruoff import NOISE_PARAMETER_IDS

## manual setting ##
problem_name = "ruoff_atp_consumption"
experiment_name = "2024_07_22_Ruoff_Grid"
analysis_name = "find_average_start"
CLUSTER = True

# read in 
filename = "ude_nrs_by_reg.txt"
ude_nrs_to_fit = [int(i) for i in np.loadtxt(Path(__file__).resolve().parent / experiment_name / filename)]

# set paths
if CLUSTER:
    storage_dir = Path("/storage/groups/hasenauer_lab/sym/").resolve()
    dir_exp_output = storage_dir / "5_optimisation" / experiment_name
    dir_sim = storage_dir / "6_evaluation" / "simulation" / experiment_name
    dir_output = storage_dir / "6_evaluation" / experiment_name / analysis_name
else:
    dir_exp_output = dir_pipeline / "5_optimisation" / experiment_name
    dir_sim = dir_exp_output / "result"
    dir_output = dir_pipeline / "6_evaluation" / experiment_name / analysis_name
dir_output.mkdir(exist_ok=True)

summary = load_exp_summary(dir_exp_output)

for model_id in ude_nrs_to_fit:
    print(model_id)
    row = summary.query("ude_nr == @model_id")
    [noise] = row["noise_level"].values
    [n_datapoints] = row["sparsity"].values
    sim = load_simulation(dir_sim, model_id)
    [nll_obs_trainval] = row["negLL_obs_trainval"].values
    [nmse_obs_test] = row["nmse_obs_test"].values

    # plot
    fig, ax = plot_fit(
        sim,
        noise,
        n_datapoints,
        predict=True,
        sd_parameters={sd: row[sd].values[0] for sd in NOISE_PARAMETER_IDS},
    )
    fig.suptitle(f"({noise}-{n_datapoints}): NLL trainval {round(nll_obs_trainval, 4)}, NMSE obs test {round(nmse_obs_test, 4)}")
    fig.tight_layout()
    fig.savefig(dir_output / f"prediction_{model_id}.svg")
    plt.close()

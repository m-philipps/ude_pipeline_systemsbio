import sys
from pathlib import Path
from helpers_ruoff import *
import pypesto
from pypesto.visualize import waterfall
from itertools import product

dir_pipeline = Path(__file__).resolve().parents[1]
dir_1 = dir_pipeline / "1_mechanistic_model"
sys.path.append(str(dir_1))
from reference_ruoff import NOISE_PARAMETER_IDS, NOISE_PERCENTAGES, DATASET_SIZES

## manual setting ##
problem_name = "ruoff_atp_consumption"
experiment_name = "2024_08_13_Ruoff_Grid"
metric = "nmae_obs_test"  # "negLL_obs_trainval", "nmse_obs_test"
metric_name = "NMAE Test"  # "Likelihood", "NMSE Test"
n_best = 50
CLUSTER = True

# set paths
if CLUSTER:
    storage_dir = Path("/storage/groups/hasenauer_lab/sym/").resolve()
    dir_exp_output = storage_dir / "5_optimisation" / experiment_name
    dir_sim = storage_dir / "6_evaluation" / "simulation" / experiment_name
    dir_output = storage_dir / "6_evaluation" / experiment_name / f"fits_by_{metric}"
else:
    dir_exp_output = dir_pipeline / "5_optimisation" / experiment_name
    dir_sim = dir_exp_output / "result"
    dir_output = dir_pipeline / "6_evaluation" / experiment_name / f"fits_by_{metric}"

summary = load_exp_summary(dir_exp_output)
# sort by the metric defined above
summary = summary.sort_values(by=metric)

for noise, ndata in product(NOISE_PERCENTAGES, DATASET_SIZES):
    subdf = summary.query("noise_level == @noise & (sparsity == @ndata)")
    if subdf.shape[0] < 1:
        continue
    print(f"NOISE {noise}, # dp {ndata}")

    dir_output_dataset = dir_output / f"{noise}_{ndata}"
    dir_output_dataset.mkdir(exist_ok=True, parents=True)

    for i_best in range(n_best):
        row = subdf.iloc[i_best, :]
        model_id = row["ude_nr"]
        sim = load_simulation(dir_sim, model_id)
        title = f"NLL trainval {round(row['negLL_obs_trainval'], 4)}, NMSE obs test {round(row['nmse_obs_test'], 4)}"
        # plot
        fig, ax = plot_fit(
            sim,
            noise,
            ndata,
            predict=True,
            sd_parameters={sd: row[sd] for sd in NOISE_PARAMETER_IDS},
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(dir_output_dataset / f"prediction_{i_best}_{model_id}.svg", transparent=False)
        plt.close()

        if i_best > 20:
            continue
        # plot full state space
        fig, ax = plot_full_state_space(
            sim,
            predict=True,
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(dir_output_dataset / f"full_space_{i_best}_{model_id}.svg", transparent=False)
        plt.close()

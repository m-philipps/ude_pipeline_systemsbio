import sys
from pathlib import Path
import json
from helpers_boehm import *
import pypesto
from pypesto.visualize import waterfall
from itertools import product

dir_pipeline = Path(__file__).resolve().parents[1]
dir_1 = dir_pipeline / "1_mechanistic_model"
sys.path.append(str(dir_1))
from reference_boehm import NOISE_PARAMETER_IDS, SPECIES_IDS


## manual setting ##
problem_name = "boehm_export_augmented"
experiment_date = "2024_11_14"
experiment_name = experiment_date + "_B" + problem_name[1:] + "_preopt"
metric = "negLL_obs_trainval"  # "negLL_obs_trainval", "nmse_obs_trainval"
metric_name = "Likelihood trainval"  # "Likelihood", "NMSE"
n_best = 12
CLUSTER = True
TRANSPARENT = True

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

with open(dir_pipeline / "problems.json", "r") as jf:
    problem = json.load(jf)[problem_name]

if "augmentation" in problem and (problem["augmentation"]["type"] == "species"):
    species_ids = list(SPECIES_IDS) + list(problem["augmentation"]["name"])
else:
    species_ids = SPECIES_IDS

summary = load_exp_summary(dir_exp_output)
# sort by the metric defined above
summary = summary.sort_values(by=metric)

dir_output.mkdir(exist_ok=True, parents=True)

# add regularisation bins
bins = [-1, 1e-4,    1e-2,  1e-1,   1,   10]
labels = ["0", "<0.01", "<0.1", "<1", "<10"]
summary["regbin"] = pd.cut(summary['λ_reg'], bins=bins, labels=labels)

for reglabel in labels:
    print(reglabel)
    summary_reg = summary.query("regbin == @reglabel")
    dir_output_reg = dir_output / reglabel.replace(".", "_").replace("<", "")
    dir_output_reg.mkdir(exist_ok=True, parents=True)

    for i_best in range(n_best):
        row = summary_reg.iloc[i_best, :]
        model_id = row["ude_nr"]
        print(model_id)
        sim = load_simulation(dir_sim, model_id)
        score = round(row[metric], 4)
        title = f"#{i_best} {metric_name} {score}"
        # plot
        fig, ax = plot_obs_fits_together(sim, {sd: row[sd] for sd in NOISE_PARAMETER_IDS})
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(dir_output_reg / f"fit_{i_best}_{model_id}.svg", transparent=TRANSPARENT)
        plt.close()
        # plot full state space
        fig, ax = plot_state_space(
            sim,
            species_ids,
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(dir_output_reg / f"state_space_{i_best}_{model_id}.svg", transparent=TRANSPARENT)
        plt.close()

    # waterfall plot
    optimize_result = pypesto.result.OptimizeResult()
    for key, val in summary_reg[metric].to_dict().items():
        single_result = pypesto.result.OptimizerResult(**{"id": str(key), "fval": val})
        optimize_result.append(single_result)
    pypesto_result = pypesto.Result(optimize_result=optimize_result)
    fig, ax = plt.subplots(figsize=(8, 5))
    waterfall(
        results=pypesto_result,
        n_starts_to_zoom=20,
        ax=ax,
    )
    ax.set_title(metric_name)
    # fig.suptitle("Waterfall: " + dataset_id.replace("_", " "))
    fig.tight_layout()
    fig.savefig(dir_output_reg / f"waterfall_{metric}.svg", transparent=TRANSPARENT)
    plt.close()

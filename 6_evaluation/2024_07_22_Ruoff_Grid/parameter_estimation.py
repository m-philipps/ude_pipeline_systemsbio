from pathlib import Path
import sys
import pandas as pd
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import pypesto 
import pypesto.visualize as vis

from helpers import load_exp_summary

dir_pipeline = Path(__file__).resolve().parents[2]
dir_1 = dir_pipeline / "1_mechanistic_model"
dir_5 = dir_pipeline / "5_optimisation"
dir_6 = Path(__file__).resolve().parents[1]
sys.path.append(str(dir_1))
sys.path.append(str(dir_6))
from helpers_ruoff import inverse_transform, load_opt_parameters

from reference_ruoff import PARAMETERS_IDS
with open(dir_pipeline / "problems.json", "r") as f:
    problem = json.load(f)["ruoff_atp_consumption"]

## manual setting ##
problem_name = "ruoff_atp_consumption"
experiment_name = "2024_07_22_Ruoff_Grid"
CLUSTER = True
noise = 20  # 5, 10, 20, 35
ndp = 200   # 50, 100, 150, 200
metric = 'nmae_obs_test'

# set paths
if CLUSTER:
    storage_dir = Path("/storage/groups/hasenauer_lab/sym/").resolve()
    dir_exp_output = storage_dir / "5_optimisation" / experiment_name
    dir_sim = storage_dir / "6_evaluation" / "simulation" / experiment_name
else:
    dir_exp_output = dir_pipeline / "5_optimisation" / experiment_name
    dir_sim = dir_exp_output / "result"

ordered_parameter_names = [p for p, i in zip(PARAMETERS_IDS, problem["mechanistic_parameters"]) if i]

def get_parameter_bounds(dir_1) -> dict:
    return pd.read_csv(dir_1 / "Ruoff_BPC2003" / "parameters_noise_5.tsv", sep="\t")[[
        "parameterId", "parameterScale", "lowerBound", "upperBound"
    ]].set_index("parameterId").to_dict()

# load data
summary = load_exp_summary(dir_exp_output)

# add reg bins
bins = [-1, 1e-4,    5e-3,    1e-2,    5e-2,   1e-1,   5e-1,   1,    5,   10]
labels = ["0", "<0.005", "<0.01", "<0.05", "<0.1", "<0.5", "<1", "<5", "<10"]
summary["regbin"] = pd.cut(summary['λ_reg'], bins=bins, labels=labels)

for noise in summary["noise_level"].unique():
    for sparsity in summary["sparsity"].unique():
        print(f"{noise}% noise - {sparsity} sparsity")
        
        df = summary.query("noise_level == @noise & (sparsity == @sparsity)").sort_values(by=metric)
        dir_output = dir_6 / experiment_name / "evaluation" / f"{noise}_{sparsity}"
        dir_output.mkdir(exist_ok=True, parents=True)

        p_scales = get_parameter_bounds(dir_1)
        
        # plot metric by reg bin
        df_best_per_reg = df.loc[df.groupby(by="regbin")[metric].idxmin().values]
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(df_best_per_reg["regbin"], df_best_per_reg[metric])
        ax.set_title(f"Best {metric} by regularisation bin")
        ax.set_xlabel("Regularisation")
        ax.set_ylabel("NMAE")
        fig.tight_layout()
        fig.savefig(dir_output / f"best_{metric}_by_reg.svg")
        
        # load PEtab parameters table and drop k_5 for UDE
        pdf = pd.read_csv(dir_1 / "Ruoff_BPC2003" / f"parameters_noise_{noise}.tsv", sep="\t")
        pdf = pdf.drop(pdf.query("parameterId == 'k_5'").index)
        
        # add a transformedValue column
        def getTransformedValue(x, scale):
            if scale == "lin":
                return x
            elif scale == "log10":
                return np.log10(x)
        
        pdf["transformedValue"] = pdf.apply(lambda row: getTransformedValue(row['nominalValue'], row['parameterScale']), axis=1)
        pdf["transformedUpperBound"] = pdf.apply(lambda row: getTransformedValue(row['upperBound'], row['parameterScale']), axis=1)
        pdf["transformedLowerBound"] = pdf.apply(lambda row: getTransformedValue(row['lowerBound'], row['parameterScale']), axis=1)
        
        # add best parameter vectors per reg to PEtab params table
        for r in labels:
            ude_nr = df.loc[df.query("regbin == @r")[metric].idxmin()]["ude_nr"]
            # load optimised parameters from julia files
            fp_p_opt = dir_exp_output / "result" / f"ude_{int(ude_nr)}" / "p_opt.jld2"
            x = load_opt_parameters(fp_p_opt)[:len(ordered_parameter_names)]
            # tranform tanh-scaled values to log scale
            for i, (xi, pid) in enumerate(zip(x, ordered_parameter_names)):
                if p_scales['parameterScale'][pid] == 'log10':
                    x[i] = np.log10(inverse_transform(xi, lb=p_scales['lowerBound'][pid], ub=p_scales['upperBound'][pid]))
            
            pdf[r] = x
        
        # plot parameter estimation error
        
        pdf_plot = deepcopy(pdf).iloc[::-1]
        redblue = [
            (1.0, 0, 0.0),
            (0.89, 0, 0.11),
            (0.78, 0, 0.22),
            (0.67, 0, 0.33),
            (0.56, 0, 0.44),
            (0.44, 0, 0.56),
            (0.33, 0, 0.67),
            (0.22, 0, 0.78),
            (0.11, 0, 0.89),
            (0.0, 0, 1.0),
        ]
        
        fig, ax = plt.subplots()
        
        ax.plot(pdf_plot["transformedValue"], pdf["parameterId"], label="True", linewidth=4, color="k")
        for r, c in zip(labels, redblue):
            ax.plot(pdf_plot[r], pdf["parameterId"], label=f"reg {r}", color=c)
        plt.plot(pdf_plot["transformedUpperBound"], pdf["parameterId"], color="k", linestyle="dashed")
        plt.plot(pdf_plot["transformedLowerBound"], pdf["parameterId"], color="k", linestyle="dashed")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dir_output / "parameters_best_by_reg.svg")
        
        # calculate sum of squares error
        square_errors = {}
        for r in labels:
            square_errors[r] = pdf[["transformedValue", r]].apply(lambda row: (row["transformedValue"] - row[r])**2, axis=1).sum()
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(labels, square_errors.values())
        ax.set_title(f"Best parameter estimation error by regularisation bin")
        ax.set_xlabel("Regularisation")
        ax.set_ylabel("Sum of squared errors")
        fig.tight_layout()
        fig.savefig(dir_output / f"best_paramest_error_by_reg.svg")


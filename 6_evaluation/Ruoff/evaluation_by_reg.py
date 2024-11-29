"Evaluate the effect of regularisation on the test loss and parameter estimation."

from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from itertools import product

import pypesto
import pypesto.visualize as vis
import pypesto.store

dir_pipeline = Path(__file__).resolve().parents[2]
dir_1 = dir_pipeline / "1_mechanistic_model"
dir_6 = Path(__file__).resolve().parents[1]
sys.path.append(str(dir_1))
sys.path.append(str(dir_6))

from helpers_ruoff import (
    load_exp_summary,
    inverse_transform,
    load_opt_parameters,
    load_simulation,
    plot_fit,
)
from reference_ruoff import (
    NOISE_PERCENTAGES,
    DATASET_SIZES,
    PARAMETERS_IDS,
    NOISE_PARAMETER_IDS,
)

problem_name = "ruoff_atp_consumption"
experiment_name_grid = "2024_07_22_Ruoff_Grid"
experiment_name_25 = "2024_08_13_Ruoff_Grid"
CLUSTER = True

# set paths
if CLUSTER:
    storage_dir = Path("/storage/groups/hasenauer_lab/sym/").resolve()
    dir_5 = storage_dir / "5_optimisation"
    dir_sim = storage_dir / "6_evaluation" / "simulation"
else:
    raise NotImplementedError

with open(dir_pipeline / "problems.json", "r") as f:
    problem = json.load(f)["ruoff_atp_consumption"]
ordered_parameter_names = [
    p for p, i in zip(PARAMETERS_IDS, problem["mechanistic_parameters"]) if i
]

# load both Ruoff experiments
summary_grid = load_exp_summary(dir_5 / experiment_name_grid)
summary_25 = load_exp_summary(dir_5 / experiment_name_25)
# combine
summary = pd.concat([summary_25, summary_grid])

# add reg bins
bins = [-1, 1e-4, 1e-2, 1e-1, 1, 10]
labels_5_bins = ["0", "<0.01", "<0.1", "<1", "<10"]
summary["regbin"] = pd.cut(summary["λ_reg"], bins=bins, labels=labels_5_bins)

# noise = 10 # 5, 10, 20, 35
# ndp = 61  # {25: 8, 50: 16, 100: 31, 150: 46, 200: 61}
metric = "nmae_obs_test"

data_settings = product(NOISE_PERCENTAGES, DATASET_SIZES)
for noise, sparsity in data_settings:
    ndp = {25: 8, 50: 16, 100: 31, 150: 46, 200: 61}[sparsity]

    print(f"\n{noise}% - {ndp}\n")

    # set dir to load optimisation results
    if ndp == 8:
        dir_exp_output = dir_5 / experiment_name_25
        dir_exp_sim = dir_sim / experiment_name_25
    else:
        dir_exp_output = dir_5 / experiment_name_grid
        dir_exp_sim = dir_sim / experiment_name_grid
    # create the output dir
    dir_output = Path(__file__).resolve().parent / f"{noise}_{ndp}"
    dir_output.mkdir(exist_ok=True, parents=True)

    df = summary.query("noise_level == @noise & (ndp == @ndp)").sort_values(by=metric)

    ## effect of regularisation on performance ##

    metric_to_plot = "nmae_obs_test"
    threshold = 0.2

    # scatter plot metric by reg.
    fig, ax = plt.subplots(
        figsize=(5, 4),
        nrows=2,
        ncols=2,
        gridspec_kw={
            "width_ratios": [1, 5],
            "height_ratios": [2, 1],
            "wspace": 0,
            "hspace": 0.05,
        },
        sharex="col",
        sharey="row",
    )
    fig.suptitle(
        f"Performance {metric} by regularisation ({noise}-{ndp})",
        fontsize="medium",
        y=0.94,
    )
    # no regularisation
    df0 = df.query("λ_reg == 0")
    ax[0, 0].scatter(df0["λ_reg"], df0[metric_to_plot], alpha=0.1, edgecolor="none")
    ax[1, 0].scatter(df0["λ_reg"], df0[metric_to_plot], alpha=0.4, edgecolor="none")
    # reg on log scale
    ax[0, 1].scatter(df["λ_reg"], df[metric_to_plot], alpha=0.1, edgecolor="none")
    ax[1, 1].scatter(df["λ_reg"], df[metric_to_plot], alpha=0.4, edgecolor="none")
    ax[1, 1].set_ylim(0.9 * df[metric_to_plot].min(), threshold)
    # scale
    for axis in ax[0, :]:
        axis.set_yscale("log")
    for axis in ax[:, 1]:
        axis.set_xscale("log")
        axis.tick_params(left=False, which="both")
    ax[1, 0].set_xticks(ticks=[0], labels=[0])
    # labeling
    ax[1, 1].set_xlabel("Regularisation strength", x=0.4)
    ax[0, 0].set_ylabel(metric_to_plot)
    # save
    fig.tight_layout()
    fig.savefig(dir_output / f"scatter_{metric}_by_reg.svg")
    plt.close()

    # bar plot good/total models
    df_ = df[df[metric_to_plot] <= threshold]

    nbins = 8
    bins = np.power(10, np.linspace(-3, 1, nbins + 1))

    fig, ax = plt.subplots(
        figsize=(5, 3),
        ncols=2,
        gridspec_kw={
            "width_ratios": [1, 5],
            "wspace": 0,
        },
        sharey="row",
    )
    fig.suptitle(
        f"Good fits ({metric} $\leq$ {threshold}) by regularisation ({noise}-{ndp})",
        fontsize="medium",
        y=0.92,
    )
    # number of starts
    ax[0].bar([0], df.query("λ_reg == 0").shape[0], color="grey", alpha=0.6)
    ax[1].hist(df["λ_reg"], bins=bins, color="grey", alpha=0.6)
    # scale
    ax[0].set_xlim(-1, 1)
    ax[0].set_xticks(ticks=[0], labels=[0])
    ax[1].set_xscale("log")
    # labeling
    ax[1].set_xlabel("Regularisation strength", x=0.4)
    ax[0].set_ylabel("# Attempts", color="dimgrey")
    # y axis colours
    ax[0].spines["left"].set_color("dimgrey")
    ax[0].tick_params(axis="y", colors="dimgrey")
    # 2nd axis: # good fits
    ax2_0 = ax[0].twinx()
    ax2_0.bar([0], df_.query("λ_reg == 0").shape[0], alpha=0.6)
    ax2_1 = ax[1].twinx()
    ax2_1.sharey(ax2_0)
    ax2_1.hist(df_["λ_reg"], bins=bins, alpha=0.6)
    # labels
    ax2_1.set_ylabel("# Good Fits", color="tab:blue")
    # y axis colours
    ax2_1.spines["right"].set_color("tab:blue")
    ax2_1.tick_params(axis="y", colors="tab:blue")
    # remove double y ticks
    ax[1].tick_params(left=False, which="both")
    ax2_0.tick_params(right=False, which="both", labelcolor="none")
    # save
    fig.tight_layout()
    fig.savefig(dir_output / f"hist_{metric}_{threshold}_by_reg.svg")
    plt.close()

    # relative bar plot
    bins = np.concatenate([[-1], np.power(10, np.linspace(-3, 1, nbins + 1))])
    labels = ["0"] + [str(b) for b in bins][1:-1]
    df["bin2plot"] = pd.cut(df["λ_reg"], bins=bins, labels=labels)
    df_ = df[df[metric_to_plot] <= threshold]
    # load all experiments
    dfsetup = pd.read_csv(dir_exp_output / "experiment_summary.csv").query(
        "noise_level == @noise & (sparsity == @sparsity)"
    )
    dfsetup["bin2plot"] = pd.cut(dfsetup["λ_reg"], bins=bins, labels=labels)

    def div_by_3(s):
        if "3" in s:
            return str(round(float(s), 3))
        else:
            return s

    # plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, df_["bin2plot"].value_counts() / dfsetup["bin2plot"].value_counts())
    fig.suptitle(
        f"Good fits ({metric} $\leq$ {threshold}) by regularisation ({noise}-{ndp})",
        fontsize="medium",
        y=0.95,
    )
    ax.set_xlabel("Regularisation strength")
    ax.set_ylabel("Relative amount of good fits")
    ax.set_xticklabels([div_by_3(s) for s in [t._text for t in ax.get_xticklabels()]])
    # save
    fig.tight_layout()
    fig.savefig(dir_output / f"hist_{metric}_{threshold}_by_reg_rel.svg")
    plt.close()

    # create and save the pypesto result object

    # get the true parameters as a reference
    def get_parameter_bounds(dir_1) -> dict:
        return (
            pd.read_csv(dir_1 / "Ruoff_BPC2003" / "parameters_noise_5.tsv", sep="\t")[
                ["parameterId", "parameterScale", "lowerBound", "upperBound"]
            ]
            .set_index("parameterId")
            .to_dict()
        )

    p_scales = get_parameter_bounds(dir_1)
    x_ref = pd.read_csv(
        dir_1 / "Ruoff_BPC2003" / f"parameters_noise_{noise}.tsv", sep="\t"
    )["nominalValue"].values
    for i, (xi, pid) in enumerate(zip(x_ref, ordered_parameter_names)):
        if p_scales["parameterScale"][pid] == "log10":
            x_ref[i] = np.log10(xi)
    ref = vis.create_references(x=x_ref, fval=1, legend="True")

    def unscale(x, param_id):
        if p_scales["parameterScale"][pid] == "log10":
            return np.log10(
                inverse_transform(
                    x, lb=p_scales["lowerBound"][pid], ub=p_scales["upperBound"][pid]
                )
            )
        elif p_scales["parameterScale"][pid] == "lin":
            return x
        else:
            raise NotImplementedError

    # create a pypesto result with fval and opt' parameters
    optimize_result = pypesto.result.OptimizeResult()
    for start_id, v in df[[metric, "ude_nr"]].T.to_dict().items():
        # load optimised parameters from julia files
        fp_p_opt = dir_exp_output / "result" / f"ude_{int(v['ude_nr'])}" / "p_opt.jld2"
        x = load_opt_parameters(fp_p_opt)[: len(ordered_parameter_names)]
        # tranform tanh-scaled values to log scale
        for i, (xi, pid) in enumerate(zip(x, ordered_parameter_names)):
            try:
                x[i] = unscale(xi, pid)
            except TypeError:
                x[i] = None
        # create result
        single_result = pypesto.result.OptimizerResult(
            **{"id": str(v["ude_nr"]), "fval": v[metric], "x": x}
        )
        optimize_result.append(single_result)
    pypesto_result = pypesto.Result(optimize_result=optimize_result)

    # use the mechanistic model's problem and fix the non-est. parameter
    # referece result from ODE
    ode_res = pypesto.store.read_result(
        dir_pipeline
        / "m_mechanistic_modelling"
        / "ruoff"
        / "output"
        / f"{noise}_100"
        / "result.hdf5",
        optimize=True,
    )
    pypesto_result.problem = ode_res.problem
    pypesto_result.problem.fix_parameters(
        [idx for idx, i in enumerate(problem["mechanistic_parameters"]) if not i], [0]
    )

    pypesto.store.write_result(
        result=pypesto_result,
        filename=dir_output / "result.hdf5",
        optimize=True,
        overwrite=True,
    )

    # parameters plot
    pypesto.visualize.parameters(
        pypesto_result, size=(8, 5), start_indices=list(range(100)), reference=ref
    )
    plt.title(f"Estimated parameters ({noise}, {ndp})")
    plt.savefig(dir_output / f"parameters_100_{metric}.svg")
    plt.close()

    # waterfall
    pypesto.visualize.waterfall(pypesto_result, size=(8, 5), n_starts_to_zoom=20)
    plt.savefig(dir_output / f"waterfall_{metric}.svg")
    plt.close()

    ## Parameter estimation error

    df_best_test_per_reg = df.loc[df.groupby(by="regbin")[metric].idxmin().values]
    df_best_train_per_reg = df.loc[
        df.groupby(by="regbin")["negLL_obs_trainval"].idxmin().values
    ]

    # best metric by regularisation bin
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df_best_test_per_reg["regbin"], df_best_test_per_reg[metric], "grey")
    ax.plot(
        df_best_test_per_reg["regbin"],
        df_best_test_per_reg[metric],
        ".k",
        markersize=10,
    )
    ax.set_title(f"Best test error by regularisation bin ({noise}, {ndp})")
    ax.set_xlabel("Regularisation")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(dir_output / f"best_{metric}_by_reg.svg")
    plt.close()

    # Construct a df with the nominal parameters, bounds, and best estimated parameters per regularisation bin, all on transformed scale.

    # load PEtab parameters table and drop k_5 for UDE
    pdf = pd.read_csv(
        dir_1 / "Ruoff_BPC2003" / f"parameters_noise_{noise}.tsv", sep="\t"
    )
    pdf = pdf.drop(pdf.query("parameterId == 'k_5'").index)

    # add a transformedValue column
    def getTransformedValue(x, scale):
        if scale == "lin":
            return x
        elif scale == "log10":
            return np.log10(x)

    pdf["transformedValue"] = pdf.apply(
        lambda row: getTransformedValue(row["nominalValue"], row["parameterScale"]),
        axis=1,
    )
    pdf["transformedUpperBound"] = pdf.apply(
        lambda row: getTransformedValue(row["upperBound"], row["parameterScale"]),
        axis=1,
    )
    pdf["transformedLowerBound"] = pdf.apply(
        lambda row: getTransformedValue(row["lowerBound"], row["parameterScale"]),
        axis=1,
    )
    # load best parameter vector per reg
    for r in df["regbin"].unique():
        for m, key in zip(["negLL_obs_trainval", "nmae_obs_test"], ["train", "test"]):
            ude_nr = df.loc[df.query("regbin == @r")[m].idxmin()]["ude_nr"]
            # load optimised parameters from julia files
            fp_p_opt = dir_exp_output / "result" / f"ude_{int(ude_nr)}" / "p_opt.jld2"
            x = load_opt_parameters(fp_p_opt)[: len(ordered_parameter_names)]
            # tranform tanh-scaled values to log scale
            for i, (xi, pid) in enumerate(zip(x, ordered_parameter_names)):
                if p_scales["parameterScale"][pid] == "log10":
                    x[i] = np.log10(
                        inverse_transform(
                            xi,
                            lb=p_scales["lowerBound"][pid],
                            ub=p_scales["upperBound"][pid],
                        )
                    )
            pdf[f"{r}_{key}"] = x

    # plot the parameter vectors by reg
    pdf_plot = deepcopy(pdf).iloc[::-1]
    redblue = [
        (1.0, 0, 0.0),
        (0.67, 0, 0.33),
        (0.44, 0, 0.56),
        (0.22, 0, 0.78),
        (0.0, 0, 1.0),
    ]
    for metric, loss in zip(["negLL_obs_trainval", "nmae_obs_test"], ["train", "test"]):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for r, c in zip(labels_5_bins, redblue):
            ax.plot(
                pdf_plot[f"{r}_{loss}"] - pdf_plot["transformedValue"],
                pdf["parameterId"],
                label=f"$\lambda$ {r}",
                color=c,
                linewidth=2,
                alpha=0.8,
            )
        plt.plot(
            pdf_plot["transformedUpperBound"] - pdf_plot["transformedValue"],
            pdf["parameterId"],
            color="k",
            linestyle="dashed",
        )
        plt.plot(
            pdf_plot["transformedLowerBound"] - pdf_plot["transformedValue"],
            pdf["parameterId"],
            color="k",
            linestyle="dashed",
        )
        ax.plot(
            pdf_plot["transformedValue"] - pdf_plot["transformedValue"],
            pdf["parameterId"],
            label="True",
            linewidth=2,
            color="k",
        )
        ax.set_xlabel("Difference to the true parameters in log space")
        ax.legend()
        fig.tight_layout()
        fig.suptitle(f"Best UDEs by {metric}", y=1.02)
        fig.savefig(
            dir_output / f"parameters_best_by_reg_relative_to_best_{loss}loss.svg"
        )

    # calculate an error for the estimated parameters; using the scaled parameters and sum of squares
    square_errors = {"train": {}, "test": {}}
    for r in df["regbin"].unique():
        square_errors["test"][r] = (
            pdf[["transformedValue", f"{r}_test"]]
            .apply(
                lambda row: (row["transformedValue"] - row[f"{r}_test"]) ** 2, axis=1
            )
            .sum()
        )
        square_errors["train"][r] = (
            pdf[["transformedValue", f"{r}_train"]]
            .apply(
                lambda row: (row["transformedValue"] - row[f"{r}_train"]) ** 2, axis=1
            )
            .sum()
        )
    # plot
    fig, ax = plt.subplots(figsize=(5, 3))
    error_ordered_train = [square_errors["train"][r] for r in labels_5_bins]
    ax.plot(labels_5_bins, error_ordered_train, "grey")
    ax.plot(labels_5_bins, error_ordered_train, ".k", markersize=10)
    ax.set_title(
        f"Parameter estimation error of best models by reg. bin ({noise}, {ndp})"
    )
    ax.set_xlabel("Regularisation")
    ax.set_ylabel("Squared error of best train model")
    ax_1 = ax.twinx()
    error_ordered_test = [square_errors["test"][r] for r in labels_5_bins]
    ax_1.plot(labels_5_bins, error_ordered_test, "tab:blue")
    ax_1.plot(labels_5_bins, error_ordered_test, ".k", markersize=10)
    ax_1.set_ylabel("Squared error of best test model", color="tab:blue")
    # y axis colours
    ax_1.spines["right"].set_color("tab:blue")
    ax_1.tick_params(axis="y", colors="tab:blue")
    fig.tight_layout()
    fig.savefig(dir_output / f"paramest_error_best_test_by_reg.svg")
    plt.close()

    # Save the best result per reg: metrics, hps settings, estimated parameters, param error
    # add estimated parameters
    for m, key, df in zip(
        ["negLL_obs_trainval", "nmae_obs_test"],
        ["train", "test"],
        [df_best_train_per_reg, df_best_test_per_reg],
    ):
        for pid in pdf["parameterId"]:
            d = (
                pdf.query("parameterId == @pid")[[l + "_" + key for l in labels_5_bins]]
                .T.iloc[:, 0]
                .to_dict()
            )
            df[pid] = df["regbin"].map(d)
        # add parameter estimation error
        df["param_est_error"] = df["regbin"].map(square_errors[key])
        # save
        df.to_csv(dir_output / f"best_{key}.csv", index=False)

    # plot some fits
    for m, key, df in zip(
        ["negLL_obs_trainval", "nmae_obs_test"],
        ["train", "test"],
        [df_best_train_per_reg, df_best_test_per_reg],
    ):
        for r, dfr in df.groupby(by="regbin"):
            ude_id = dfr["ude_nr"].values[0]
            dir_output_reg = dir_output / f"fit_{r}"
            dir_output_reg.mkdir(exist_ok=True)
            sim = load_simulation(
                experiment_output_path=dir_exp_sim,
                model_id=ude_id,
            )
            if sim is None:
                print(f"\n#{ude_id} simulation not available.\n")
                continue
            title = f"#{ude_id} - NLL {round(dfr['negLL_obs_trainval'].values[0], 3)}, NMAE {round(dfr['nmae_obs_test'].values[0], 3)}"
            # plot
            fig, ax = plot_fit(
                sim,
                noise,
                sparsity,
                True,
                {sd: dfr[sd].values[0] for sd in NOISE_PARAMETER_IDS},
            )
            # remove legends
            ax[0].get_legend().remove()
            ax[1].get_legend().remove()
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(dir_output_reg / f"best_{key}_{ude_id}.svg", transparent=False)
            plt.close()

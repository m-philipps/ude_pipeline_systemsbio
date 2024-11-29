from pathlib import Path
import matplotlib.pyplot as plt
import pypesto
import pypesto.petab

from _hyperparameters import pairs
from load_problem import petab_problems, pypesto_problems
from utils import *

# path handling
dir_output = Path(__file__).resolve().parent / "output"
dir_eval = Path(__file__).resolve().parent / "evaluation"

# remove the 25er data sets
pairs = [p for p in pairs if not p[1] == 25]
# load results
results = {
    "%d_%d"
    % pair: pypesto.store.read_result(
        dir_output / ("%d_%d" % pair) / "result.hdf5",
        optimize=True,
    )
    for pair in pairs
}

for pair_id in results.keys():
    (dir_eval / pair_id).mkdir(exist_ok=True, parents=True)


# calculate the test loss for each result
for pair_id, result in results.items():
    noise, n_dp = pair_id.split("_")
    pypesto_problem = pypesto_problems[int(noise)][int(n_dp)]
    for res in result.optimize_result:
        p_opt = res.x
        if p_opt is None:
            res["nmse_test"] = None
            continue
        # calculate test losses
        res["nmse_test"] = NMSE_observables(p_opt, pypesto_problem)
# plot fval and nmse
for pair_id, result in results.items():
    noise, n_dp = pair_id.split("_")
    fp = dir_eval / pair_id / "fvals_nmses.svg"
    fig, ax = plot_fval_nmse(result, noise, n_dp)
    fig.savefig(fp)
    plt.close()


# load reference
reference = get_reference()

# waterfall, parameters plots
for pair_id, result in results.items():
    pypesto.visualize.waterfall(result, size=(8, 5))
    plt.savefig(dir_eval / pair_id / "waterfall.svg")
    plt.close()
    pypesto.visualize.parameters(result, size=(8, 5))
    plt.savefig(dir_eval / pair_id / "parameters.svg")
    plt.close()

# simulation and trajectories
for pair_id, result in results.items():
    noise, n_dp = pair_id.split("_")
    # best fit
    petab_problem = petab_problems[int(noise)][int(n_dp)]
    pypesto_problem = pypesto_problems[int(noise)][int(n_dp)]
    fp = dir_eval / pair_id / "prediction_noise.svg"
    fig, ax = plot_fit(
        petab_problem, pypesto_problem, result.optimize_result[0].x, noise, n_dp, plot_noise=True
    )
    fig.savefig(fp)
    plt.close()

# compare the best losses
best_losses = {
    key: [
        res.optimize_result[0].fval,
        res.optimize_result[0].nmse_test,
    ]
    for key, res in results.items()
}
df_losses = pd.DataFrame(best_losses, index=["training", "test"]).T.reset_index(
    names="pair"
)
df_losses[["noise", "datapoints"]] = (
    df_losses["pair"].str.split("_", expand=True).astype(int)
)
df_losses.drop(columns=["pair"], inplace=True)
# for visualisation, normalise w.r.t. the number of measurements
df_losses["training_normalised"] = df_losses["training"] / df_losses["datapoints"]

# heatmap training losses
ax = sns.heatmap(
    df_losses.pivot(index="noise", columns="datapoints", values="training_normalised"),
    annot=True,
    fmt=".2f",
    cmap="Blues",
)
ax.set(xlabel="Datapoints [#]", ylabel="Noise [%]")
ax.set_title("Best NLLH from training")
fp_output = dir_eval / "heatmap_training.svg"
ax.figure.savefig(fp_output)
plt.close()

# heatmap test losses
ax = sns.heatmap(
    df_losses.pivot(index="noise", columns="datapoints", values="test"),
    annot=True,
    fmt=".2f",
    cmap="Greens",
)
ax.set(xlabel="Datapoints [#]", ylabel="Noise [%]")
ax.set_title("Best NMSE from test")
fp_output = dir_eval / "heatmap_test.svg"
ax.figure.savefig(fp_output)
plt.close()

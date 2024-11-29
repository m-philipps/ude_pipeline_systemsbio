import sys
from pathlib import Path
import pypesto
import fides

from _hyperparameters import NOISE, N_DP
from utils import NMSE_observables

# load problems description, reference
dir_pipeline = Path(__file__).resolve().parents[2]
sys.path.append(str(dir_pipeline))

# problems
from load_problem import petab_problems, pypesto_problems

petab_problem = petab_problems[NOISE][N_DP]
pypesto_problem = pypesto_problems[NOISE][N_DP]

# output
dir_output = Path(__file__).parent.resolve() / "output" / f"{NOISE}_{N_DP}"
dir_output.mkdir(parents=True, exist_ok=True)

optimizer = pypesto.optimize.FidesOptimizer(
    verbose=0,
    hessian_update=fides.BFGS(),
    options={"maxiter": 10000},
)

result = pypesto.optimize.minimize(
    problem=pypesto_problem,
    optimizer=optimizer,
    n_starts=200,
)

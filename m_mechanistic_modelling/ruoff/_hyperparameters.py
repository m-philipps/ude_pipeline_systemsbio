import os
import sys
from pathlib import Path
import json
import itertools

dir_pipeline = Path(__file__).resolve().parents[2]
dir_1 = dir_pipeline / "1_mechanistic_model"
for d in [dir_pipeline, dir_1]:
    sys.path.append(str(d))

from reference_ruoff import (
    NOISE_PERCENTAGES,
    DATASET_SIZES,
)

# iterate over all key pairs
pairs = [t for t in list(itertools.product(NOISE_PERCENTAGES, DATASET_SIZES))]
print(len(pairs))

# get settings
slurm_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

NOISE = pairs[slurm_id][0]
N_DP = pairs[slurm_id][1]

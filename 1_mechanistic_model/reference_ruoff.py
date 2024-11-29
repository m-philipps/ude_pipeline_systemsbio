"Set a reference for properties of the Ruoff model."

model_name_petab = "Ruoff_BPC2003"
model_name = "ruoff"

# parameters
PARAMETERS_IDS = (
    "J_0",
    "k_1",
    "k_2",
    "k_3",
    "k_4",
    "k_5",
    "k_6",
    "k_ex",
    "kappa",
    "K_1",
    "N",
    "A",
    "phi",
    # noise parameters
    "sd_N_2_obs",
    "sd_A_3_obs",
)
NOISE_PARAMETER_IDS = ["sd_N_2_obs", "sd_A_3_obs"]

# species
SPECIES_IDS = ("S_1", "S_2", "S_3", "S_4", "N_2", "A_3", "S_4_ex")
SPECIES_NAMES_SHORT = ("S_1", "S_2", "S_3", "S_4", "N_2", "A_3", "S_{4, ex}")
SPECIES_NAMES = (
    "Glucose",
    "Glyceraldehyde-3-Pydihydroxyacetone-P",
    "1,3-Bisphosphoglycerate",
    "Pyruvateyacetaldehyde",
    "NADH",
    "ATP",
    "External pyruvateyacetaldehyde",
)
OBSERVED = ["N_2", "A_3"]
# initial condition
SPECIES_IC = [1.187, 0.193, 0.05, 0.115, 0.077, 2.475, 0.077]

# observables
OBSERVABLES_IDS = ("N_2_obs", "A_3_obs")
OBSERVABLES_NAMES = ("NADH", "ATP")
OBSERVABLE_SPECIES_IDS = [4, 5]

# characterise synthetic data
DATASET_SIZES = (25, 50, 100, 150, 200)
NOISE_PERCENTAGES = (5, 10, 20, 35)
TRAINING_ENDPOINT = 1.5
TEST_ENDPOINT = 5

"Set a reference for properties of the Boehm model."

model_name_petab = "Boehm_JProteomeRes2014"
model_name = "boehm"

# parameters
PARAMETERS_IDS = (
    "Epo_degradation_BaF3",
    "k_imp_homo",
    "k_imp_hetero",
    "k_exp_homo",
    "k_exp_hetero",
    "k_phos",
    # noise parameters
    "sd_pSTAT5A_rel",
    "sd_pSTAT5B_rel",
    "sd_rSTAT5A_rel",
)
PARAMETERS_IDS_AUGMENTATION = (
    "k_exp_aug_A",
    "k_exp_aug_AB",
    "k_exp_aug_B",
)
NOISE_PARAMETER_IDS = ["sd_pSTAT5A_rel", "sd_pSTAT5B_rel", "sd_rSTAT5A_rel"]

# species
SPECIES_IDS = (
    "STAT5A",
    "STAT5B",
    "pApA",
    "pApB",
    "pBpB",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
)
# initial condition
SPECIES_IC = [143.867, 63.733, 0, 0, 0, 0, 0, 0]

# observables
OBSERVABLES_IDS = (
    "pSTAT5A_rel",
    "pSTAT5B_rel",
    "rSTAT5A_rel",
)

T_START = 0
T_END = 240

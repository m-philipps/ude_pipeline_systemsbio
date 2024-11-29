# UDE Pipeline for parameter estimation in systems biology

Contains the code to reproduce the analysis for the paper **Universal differential equations for systems biology: Current state and open problems**.


## Repo structure
- `problems.json`: Each problem is listed with a short description and the interface between the mechanistic and ANN model components
- `1_mechanistic_model`: SBML, PEtab problem for each mechanistic, non-UDE model
- `2_measurements`: Store synthetic data; Funtions for data generation and loading
- `3_start_points`: Generate start points ($\theta_M$ and $\theta_{\text{ANN}}$) for optimisation, and sample hyperparameters for the UDE optimisation
- `4_ude`: Define the UDE model
- `5_optimisation`: Hyperparameter settings, scripts to run optimisation on cluster, optimisation results
- `6_evaluate`: Evaluation and visualisation of the results
- `m_mechanistic_modeling`: Reference results for the full ODE models

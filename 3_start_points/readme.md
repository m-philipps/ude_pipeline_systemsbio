This folder contains the start points to be used for each UDE optimisation problem.

### Mechanistic parameters

Each subfolder (corresp. to an ODE or UDE problem) contains the start points for $\theta_M$, e.g. `startpoints_lhs.csv`
- `lhs` or `preopt` refers to the sampling strategy
    - LHS: latin hypercube sampling
    - PREOPT: taken from prior optimisation of only the ODE model
- suffix `noise_x` indicates the noise level in the Ruoff data


### ANN parameters

Within each subfolder (corresp. to an ODE or UDE problem) there are start points for $\theta_{ANN}$. These were sampled individually for each FNN width and depth. All bias weights and the weights of the last layer were initialised to be $0$. The files are named `fnn_{width}_{depth}.json`.

### ANN hyperparameters

Within each subfolder the `hps.csv` provides the settings for the ANN and optimisation hyperparameters. A subset of:
- `hidden_layers`: Number of ANN layers
- `hidden_neurons`: Number of neurons/layer
- `act_fct`: Activation function
- `nn_input_normalization`: Binary, ANN input normalisation
- `λ_reg`: Regularisation strength of the weight decay
- `lr_adam`: Initial learning rate of the ADAM optimiser

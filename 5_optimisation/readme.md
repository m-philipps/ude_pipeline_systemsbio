This folder contains code to run the optimization and the corresponding results. 

### General Structure

To start the optimization pipeline`run_experiment.jl` is called after setting the respective `experiment_name`. Once the optimiation is finished, `create_array_summary.jl` can be called to generate `experiment_summary.csv` - a problem specific CSV file summarizing key metrics of the optimisation result per UDE.

### Experiment Definition & Results
Each subfolder corresponds to one experiment (e.g. Scenario 1 with a specific startpoint and hyperparameter setting). This folder contains the file `experiment_definition.jl`, which defines the general settings of the problem. Based on this file, the optimisation routine can be started, i.e. `run_experiment.jl` can be called. 
The output of the UDE optimization routine is stored alongside the experiment definition:
* One model folder per UDE, containing the model's parameters, the metrics of the loss curve and the reported training times and number of epochs of this UDE
* `experiment_overview.csv` is the CSV file reporting on the data setting, startpoint number and hyperparameter setting for each UDE number
* `experiment_summary.csv` 


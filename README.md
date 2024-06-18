# SmartPL: An Integrated Approach for Platoons Driving on Mixed-Traffic Freeways

## File Description

```
├─.vscode
├─coop -> CoOP baseline functions
├─envs 
│  ├─cfg -> environment settings
│  └─ highway_gym.py ->freeway simulation environment
│  └─ platoon.py ->platoon plugin based on Plexe
├─models
│  └─ __init__.py -> import MaskablePPO
│  └─ custom_feature_extractor.py ->defined GNN,CNN, and MLP based feature extractor
├─noisy-madqn -> Noisy-MADQN baseline
│  └─ env -> simulation environment based on MADQN settings
│  └─ evaluate_freeways.py -> evaluate MADQN on different conditions
├─results -> figures in SmartPL
│  
├─ callbacks.py -> saving best model and hyperparameters during training
├─ ablation_safe_monitor.py -> ablation study on safe monitor
├─ evaluate.py -> evaluate SmartPL, CoOP, and Plexe on different conditions
├─ config.yaml -> hyperparameters settings on environment and models
├─ experiment.py -> init environment and model settings and begin training
├─ requirements.txt -> needed python moudule
├─ SmartPL_Plot.py -> the function of plot in SmartPL
├─ utils.py -> used function in SmartPL_Plot
├─ main.py -> training or testing RL model
```

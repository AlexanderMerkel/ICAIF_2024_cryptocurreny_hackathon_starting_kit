# %%
import pickle
import numpy as np
from statsmodels.tsa.api import VAR
from arch import arch_model


# Load the training data
with open("./data/ref_log_return.pkl", "rb") as f:
    data = pickle.load(f)
data = np.array(data)  # Shape [days, hours, assets] = [8937, 24, 3]
print("Training data shape:", data.shape)

# Load your data (assuming it is stored in the variable 'data')
reshaped_data = data.reshape(-1, 3)  # Reshape to 2D: (time points * 24, 3)

# Fit a VAR model to capture dependencies between the series
var_model = VAR(reshaped_data)
var_result = var_model.fit(maxlags=1, ic='aic')

# Extract residuals from the VAR model for volatility modeling
residuals = var_result.resid

# Fit univariate GARCH(1,1) models to each residual series
garch_models = [
    arch_model(residuals[:, i], vol='Garch', p=1, q=1)
    for i in range(residuals.shape[1])
]

# Fit the GARCH models
fitted_garch_models = [model.fit(disp='off') for model in garch_models]

# Save the fitted GARCH models
with open("models/garch_model.pkl", "wb") as f:
    pickle.dump(fitted_garch_models, f)
# %%
def generate_samples(n_samples):
    # Load the fitted GARCH models
    with open("models/garch_model.pkl", "rb") as f:
        fitted_garch_models = pickle.load(f)
    
    samples = []
    for model in fitted_garch_models:
        # Generate samples from the GARCH model
        sim_data = model.model.simulate(model.params, n_samples)
        samples.append(sim_data['data'])
    
    # Reshape the samples to match the input data structure
    samples = np.column_stack(samples)
    return samples


# %% Evaluate the model
import ml_collections
import copy
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from os import path as pt
import pickle
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.summary import full_evaluation
from src.utils import set_seed, save_obj, load_obj
def evaluate_model():
    with open("./data/ref_log_return.pkl", "rb") as f:
        loaded_array = pickle.load(f)
    train_log_return = torch.tensor(loaded_array)
    print(train_log_return.shape)

    with open("./data/ref_price.pkl", "rb") as f:
        loaded_array = pickle.load(f)
    train_init_price = torch.tensor(loaded_array)
    print(train_init_price.shape)
    ### Generative models for time series generation
    # Load configuration dict
    config_dir = 'configs/config.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
        
    set_seed(config.seed)

    config.update({"device": "cpu"}, allow_val_change=False)
        
    class XYDataset(TensorDataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
            self.shape = X.shape

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

    ### Data Construction
    perm_idx = torch.randperm(train_log_return.shape[0])
    train_size = int(0.8*train_log_return.shape[0])

    cv_training_data = train_log_return[perm_idx[:train_size]].to(config.device).to(torch.float)
    cv_init_price = train_init_price[perm_idx[:train_size]].to(config.device).to(torch.float)
    cv_validation_data = train_log_return[perm_idx[train_size:]].to(config.device).to(torch.float)
    cv_val_init_price = train_init_price[perm_idx[train_size:]].to(config.device).to(torch.float)

    # Load the dataset
    training_set = TensorDataset(cv_init_price, cv_training_data)

    train_dl = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    config.input_dim = cv_training_data[0][0].shape[-1]

    ### Generative model
    from src.baselines.networks.discriminators import Discriminator
    from src.baselines.networks.generators import Generator
    ### Initialize the generator, discriminator and the trainer
    generator = Generator(config)
    discriminator = Discriminator(config)

    ### Model training and saving
    ### Synthetic data generation
    ### Model evaluation
    from src.evaluation.strategies import log_return_to_price

    fake_data = torch.tensor(generate_samples(24 * 3 * 1800)).float()

    config_dir = 'src/evaluation/config.yaml'
    with open(config_dir) as file:
        eval_config = ml_collections.ConfigDict(yaml.safe_load(file))

    # Ensure eval_size is defined
    eval_size = min(len(fake_data), len(cv_val_init_price))

    fake_prices = log_return_to_price(fake_data[:eval_size], cv_val_init_price[:eval_size])
    cv_val = log_return_to_price(cv_validation_data[:eval_size], cv_val_init_price[:eval_size])

    all_positive = (fake_prices > 0).all()
    if not all_positive:
        raise ValueError("Sanity Check Failed: Some fake prices are not positive.")

    res_dict = {"var_mean" : 0., "es_mean": 0., "max_drawback_mean": 0., "cumulative_pnl_mean": 0.,}

    # Do final evaluation
    num_strat = 4
    with torch.no_grad():
        for strat_name in ['equal_weight', 'mean_reversion', 'trend_following', 'vol_trading']:
            subres_dict = full_evaluation(fake_prices, cv_val, eval_config, strat_name = strat_name)
            for k in res_dict:
                res_dict[k] += subres_dict[k] / num_strat
            
    for k, v in res_dict.items():
        print(k, v)
# %%
import pickle
import numpy as np
from statsmodels.tsa.api import VAR, ARIMA

# Load the training data
with open("./data/ref_log_return.pkl", "rb") as f:
    data = pickle.load(f)
data = np.array(data)  # Shape [days, hours, assets] = [8937, 24, 3]
print("Training data shape:", data.shape)

# Reshape the data to 2D for the VAR model
reshaped_data = data.reshape(-1, 3)

# Fit a VAR model to capture dependencies between the series
var_model = VAR(reshaped_data)
var_result = var_model.fit(maxlags=1, ic='aic')

# Extract residuals from the VAR model for further time series modeling
residuals = var_result.resid

# Fit univariate ARIMA models to each residual series
arima_models = [
    ARIMA(residuals[:, i], order=(1, 1, 1))
    for i in range(residuals.shape[1])
]

# Fit the ARIMA models
fitted_arima_models = [model.fit() for model in arima_models]

# Save the fitted ARIMA models
with open("models/arima_model.pkl", "wb") as f:
    pickle.dump(fitted_arima_models, f)
# %%
def generate_samples(n_samples):
    # Load the fitted ARIMA models
    with open("models/arima_model.pkl", "rb") as f:
        fitted_arima_models = pickle.load(f)
    
    samples = []
    for model in fitted_arima_models:
        # Generate samples from the ARIMA model using forecast
        sim_data = model.forecast(steps=n_samples)
        samples.append(sim_data)

    # Reshape the samples to match the input data structure
    samples = np.column_stack(samples)
    return samples

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Load the training data
with open("./data/ref_log_return.pkl", "rb") as f:
    loaded_array = pickle.load(f)
train_log_return = np.array(loaded_array)  # Shape [days, hours, assets] = [8937, 24, 3]
print("Training data shape:", train_log_return.shape)

# %%
# Plot the correlation matrix of the log-returns

plt.figure(figsize=(8, 6))
sns.heatmap(np.corrcoef(train_log_return.reshape(-1, 3).T), annot=True, fmt=".2f", cmap="coolwarm", xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])
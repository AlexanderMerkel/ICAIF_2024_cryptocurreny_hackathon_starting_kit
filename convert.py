import pickle
import pandas as pd
import numpy as np

# Define the path to your .pkl file
file_path = "C:/Users/alex_/OneDrive/Dokumente/Repos/ICAIF_2024_cryptocurreny_hackathon_starting_kit/data/ref_log_return.pkl"

# Load the data from the pickle file
with open(file_path, "rb") as file:
    log_returns = pickle.load(file)

# Check the shape of the loaded data
print(f"Original shape: {log_returns.shape}")

# Reshape the data to 2D (assuming the third dimension represents different variables)
# shape=(8937, 24, 3) and convert it to shape=(8937x24, 3)
log_returns_reshaped = log_returns.reshape(-1, log_returns.shape[-1])

# Convert the reshaped data to a pandas DataFrame
log_returns_df = pd.DataFrame(log_returns_reshaped)

# Define the path to save the CSV file
csv_file_path = "C:/Users/alex_/OneDrive/Dokumente/Repos/ICAIF_2024_cryptocurreny_hackathon_starting_kit/data/ref_log_return.csv"

# Save the DataFrame as a CSV file
log_returns_df.to_csv(csv_file_path, index=False)


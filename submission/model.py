# %%
import pickle
import torch
import torch.nn as nn

class GARCHGenerator(nn.Module):
    def __init__(self, coefficients):
        super(GARCHGenerator, self).__init__()
        self.coefficients = coefficients
        print(coefficients)

    def forward(self, batch_size, device):
        params = self.coefficients
        # params is a tensor of shape 3x4 with asset x (mu, omega, alpha, beta)
        sim_data = simulate_garch_paths(params, 24 * batch_size, device)
        sim_data = sim_data.view(batch_size, 24, 3)
        return torch.tensor(sim_data, dtype=torch.float32).to(device)

def simulate_garch_paths(params, n, device="cpu", distribution="normal"):
    # unpack paras to four separate tensors
    mu = params[:, 0]
    omega = params[:, 1]
    alpha = params[:, 2]
    beta = params[:, 3]
    
    paths = torch.zeros(3, n, device=device)
    variances = torch.zeros(3, n, device=device)
    for t in range(1, n):
        variances[:, t] = omega + alpha * paths[:, t-1]**2 + beta * variances[:, t-1]
        if distribution == "normal":
            noise = torch.normal(0, torch.sqrt(variances[:, t]))
        elif distribution == "t":
            noise = torch.distributions.StudentT(df=5).rsample(variances[:, t].shape) * torch.sqrt(variances[:, t])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        paths[:, t] = mu + noise
    return paths

def load_parameters(device="cpu"):
    with open("model_dict.pkl", "rb") as f:
        coefficients = pickle.load(f)
    coefficients = torch.tensor(coefficients, device=device)
    return coefficients

def init_generator(device="cpu"):
    coefficients = load_parameters(device)
    generator = GARCHGenerator(coefficients)
    return generator

generator = init_generator()

# %%
# if __name__ == "__main__":
#     batch_size = 1800  # Generate 1800 samples of data (24x3)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     generator = init_generator(device)
#     fake_data = generator(batch_size, device)
#     print(fake_data.shape, fake_data.type())
#     # Save the fake data
#     with open("fake_log_return.pkl", "wb") as f:
#         pickle.dump(fake_data.cpu(), f)
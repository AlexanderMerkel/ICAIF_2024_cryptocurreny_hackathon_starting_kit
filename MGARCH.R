# Load the required packages
if (!require("rmgarch")) install.packages("rmgarch", dependencies=TRUE)
library(rmgarch)

# Define the path to your .csv file
file_path <- "C:/Users/alex_/OneDrive/Dokumente/Repos/ICAIF_2024_cryptocurreny_hackathon_starting_kit/data/ref_log_return.csv"

# Load the data
log_returns <- read.csv(file_path)

# Convert to an R matrix (assuming log_returns is structured as a T x d array)
log_returns <- as.matrix(log_returns)

# Define the dimensions
d <- ncol(log_returns)  # Number of dimensions (variables)

# Specify the univariate GARCH model
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = "norm")

# Create a multivariate GARCH specification
multispec <- multispec(replicate(d, spec))

# Fit the DCC-GARCH(1,1) model using the rmgarch package
dcc_spec <- dccspec(uspec = multispec, dccOrder = c(1, 1), distribution = "mvnorm")
print(dcc_spec)
# Fit the model to the log-return data with verbose output
dcc_fit <- dccfit(dcc_spec, data = log_returns, fit.control = list(eval.se = TRUE), verbose = TRUE)

# Display the fitted model summary
summary(dcc_fit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset and determine the number of rows and columns
data = pd.read_csv("/Users/aayush./Desktop/Numerical/real_estate_dataset.csv")

rows, cols = data.shape

# Extract column headers
col_headers = data.columns

# Save the column names to a text file
np.savetxt("headers_list.txt", col_headers, fmt="%s")

# Select specific features for the predictor variables
features = data[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]]

# Choose 'Price' as the target variable
target = data["Price"].values

# Determine the dimensions of the feature matrix
n_rows, n_features = features.shape

# Initialize coefficient array, including the intercept, with ones
initial_weights = np.ones(n_features + 1)

# Compute predictions without modifying the feature matrix
def_predictions = features @ initial_weights[1:] + initial_weights[0]

# Append a column of ones to represent the intercept term
features_with_bias = np.hstack((np.ones((n_rows, 1)), features))

# Compute predictions using the bias-included feature matrix
bias_predictions = features_with_bias @ initial_weights

# Check if predictions match using both methods
same_predictions = np.allclose(def_predictions, bias_predictions)
print(same_predictions)

# Compute residuals and errors using predictions and actual values
residuals = target - bias_predictions
relative_residuals = residuals / target

# Calculate mean squared error using iteration
mse_loop = 0
for idx in range(n_rows):
    mse_loop += residuals[idx] ** 2
mse_loop /= n_rows

# Calculate mean squared error using matrix multiplication
mse_matrix = np.transpose(residuals) @ residuals / n_rows

# Verify both methods yield similar results for MSE
mse_equal = np.allclose(mse_loop, mse_matrix)
print(f"MSE computed via loop and matrix methods match: {mse_equal}\n")

# Print the residuals array dimensions and their L2 norm
print(f"Residual size: {residuals.shape}")
print(f"L2 norm of residuals: {np.linalg.norm(residuals)}")
print(f"L2 norm of relative residuals: {np.linalg.norm(relative_residuals)}")

# Recompute MSE using matrix-based operations for consistency
mse_matrix = (target - features_with_bias @ initial_weights).T @ (target - features_with_bias @ initial_weights) / n_rows

# Derive the gradient of MSE with respect to coefficients
grad_mse = -2 / n_rows * features_with_bias.T @ (target - features_with_bias @ initial_weights)

# Solve normal equations to optimize coefficients
optimized_weights = np.linalg.inv(features_with_bias.T @ features_with_bias) @ features_with_bias.T @ target

# Save optimized coefficients to a file
np.savetxt("optimized_weights.csv", optimized_weights, delimiter=",")

# Calculate predictions and residuals using the optimized model
final_predictions = features_with_bias @ optimized_weights
final_residuals = target - final_predictions
print(f"L2 norm of final residuals: {np.linalg.norm(final_residuals)}")
final_relative_residuals = final_residuals / target

# Exclude 'Price' column to create the feature matrix without labels
features_extracted = data.drop("Price", axis=1).values
target_extracted = data["Price"].values

# Re-determine matrix dimensions after column extraction
n_rows, n_features = features_extracted.shape

# Include the bias column in the features matrix
features_with_bias = np.hstack((np.ones((n_rows, 1)), features_extracted))

# Calculate the matrix rank of the Gramian matrix
gramian_rank = np.linalg.matrix_rank(features_with_bias.T @ features_with_bias)

# Apply QR factorization to decompose the matrix
q_matrix, r_matrix = np.linalg.qr(features_with_bias)

# Save the upper triangular matrix to a CSV file
np.savetxt("upper_triangular_r.csv", r_matrix, delimiter=",")

# Use QR decomposition for coefficient derivation
b_vector = q_matrix.T @ target_extracted
coeff_qr = np.linalg.inv(r_matrix) @ b_vector

# Solve system of equations using backward substitution
coeff_qr_backward = np.zeros(n_features + 1)
for idx in range(n_features, -1, -1):
    coeff_qr_backward[idx] = b_vector[idx]
    for j in range(idx + 1, n_features + 1):
        coeff_qr_backward[idx] -= r_matrix[idx, j] * coeff_qr_backward[j]
    coeff_qr_backward[idx] /= r_matrix[idx, idx]

# Save results of backward substitution to a file
np.savetxt("qr_backward_coefficients.csv", coeff_qr_backward, delimiter=",")

# Perform Singular Value Decomposition (SVD)
U, Sigma, V_trans = np.linalg.svd(features_with_bias)

# Construct the pseudo-inverse of the singular value matrix
sigma_inv = np.zeros((V_trans.shape[0], U.shape[0]))
np.fill_diagonal(sigma_inv, 1 / Sigma)

# Calculate the Moore-Penrose pseudo-inverse of the feature matrix
pseudo_inverse = V_trans.T @ sigma_inv @ U.T

# Derive coefficients using the pseudo-inverse
coeff_svd = pseudo_inverse @ target_extracted

# Save SVD-derived coefficients to a file
np.savetxt("coefficients_svd.csv", coeff_svd, delimiter=",")

# Compute predictions and residuals using SVD-based coefficients
svd_predictions = features_with_bias @ coeff_svd
svd_residuals = target_extracted - svd_predictions

# Compute L2 norms for residuals and save them to files
print(f"L2 norm of residuals (SVD): {np.linalg.norm(svd_residuals)}")
np.savetxt("svd_residuals.csv", svd_residuals, delimiter=",")

svd_relative_residuals = svd_residuals / target_extracted
np.savetxt("relative_residuals_svd.csv", svd_relative_residuals, delimiter=",")
print(f"L2 norm of relative residuals (SVD): {np.linalg.norm(svd_relative_residuals)}")

# Perform eigenvalue decomposition of the Gramian matrix
eigen_vals, eigen_vecs = np.linalg.eig(features_with_bias.T @ features_with_bias)

# Save eigenvalues and eigenvectors to separate files
np.savetxt("eigen_values.csv", eigen_vals, delimiter=",")
np.savetxt("eigen_vectors.csv", eigen_vecs, delimiter=",")

# Reconstruct the Gramian matrix using eigen decomposition
reconstructed_gramian = eigen_vecs @ np.diag(eigen_vals) @ eigen_vecs.T

# Verify reconstruction accuracy
accurate_reconstruction = np.allclose(features_with_bias.T @ features_with_bias, reconstructed_gramian)
print(f"Gramian reconstruction accurate: {accurate_reconstruction}")

# Save the reconstructed Gramian matrix
np.savetxt("reconstructed_gramian.csv", reconstructed_gramian, delimiter=",")

# Plotting Price vs. Square Feet with regression line
square_feet = features["Square_Feet"]
intercept = optimized_weights[0]
slope = optimized_weights[1]  # Coefficient for Square_Feet

# Regression line
regression_line = intercept + slope * square_feet

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(square_feet, target, color="blue", label="Data Points")
plt.plot(square_feet, regression_line, color="red", label="Regression Line", linewidth=2)
plt.title("Price vs. Square Feet with Regression Line", fontsize=16)
plt.xlabel("Square Feet", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb

# Generalized Exponential Function
def gen_exp_function(distance, eta, kappa):
    return np.exp(-(distance / eta) ** kappa)

# Generalized Lorentz Function
def gen_lorentz_function(distance, eta, nu):
    return 1 / (1 + (distance / eta) ** nu)

# Atomic Rigidity Index Matrix Calculation
def atomic_rigidity_index_matrix(coordinates, eta, kernel_function, kernel_params):
    n_atoms = coordinates.shape[0]
    rigidity_matrix = np.zeros((n_atoms, n_atoms))

    # Calculate pairwise distances using cdist
    pairwise_distances = cdist(coordinates, coordinates)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                distance = pairwise_distances[i, j]
                rigidity_matrix[i, j] = kernel_function(distance, eta, *kernel_params)

    return rigidity_matrix




# Atomic Flexibility Index
def atomic_flexibility_index(rigidity_indices):
    flexibility_indices = 1.0 / np.maximum(rigidity_indices, np.finfo(float).eps)
    return flexibility_indices

# B-factor Prediction
def predict_b_factors(flexibility_indices, a, b):
    b_factors = a * flexibility_indices + b
    return b_factors

# Load PDB file
ppdb = PandasPdb().read_pdb('1e6j.pdb')
atom_df = ppdb.df['ATOM']
coordinates = atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()

# Parameters for correlation kernels
kappa = 2
nu = 2
eta = 5.0

# Calculate rigidity matrices
rigidity_matrix_exp = atomic_rigidity_index_matrix(coordinates, eta, gen_exp_function, (kappa,))
rigidity_matrix_lorentz = atomic_rigidity_index_matrix(coordinates, eta, gen_lorentz_function, (nu,))

# Calculate flexibility indices
flexibility_indices_exp = atomic_flexibility_index(rigidity_matrix_exp)
flexibility_indices_lorentz = atomic_flexibility_index(rigidity_matrix_lorentz)

# Example constants for B-factor prediction
a = 1.0
b = 0.0

# Predict B-factors
b_factors_exp = predict_b_factors(flexibility_indices_exp, a, b)
b_factors_lorentz = predict_b_factors(flexibility_indices_lorentz, a, b)

# Calculate Flexibility Hessian Matrix (placeholder function)
# hessian_matrix_exp = calculate_flexibility_hessian(coordinates, flexibility_indices_exp)
# hessian_matrix_lorentz = calculate_flexibility_hessian(coordinates, flexibility_indices_lorentz)

# Example CSV output (you may adjust as needed)
df_exp = pd.DataFrame(rigidity_matrix_exp)
df_lorentz = pd.DataFrame(rigidity_matrix_lorentz)

df_exp.to_csv('rigidity_indices_exponential.csv', index=False)
df_lorentz.to_csv('rigidity_indices_lorentz.csv', index=False)

print("Rigidity and flexibility indices have been saved to CSV files.")

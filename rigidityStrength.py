import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt




# Generalized Exponential Function
def gen_exp_function(distance, eta, kappa):
    return np.exp(-(distance / eta) ** kappa)


# Generalized Lorentz Function
def gen_lorentz_function(distance, eta, nu):
    return 1 / (1 + (distance / eta) ** nu)


    
# Atomic Rigidity Index Matrix Calculation
def atomic_rigidity_index_matrix(coordinates, eta, kernel_function, kernel_params):
    # Calculate pairwise distances using cdist
    pairwise_distances = cdist(coordinates, coordinates)

    # Apply the kernel function to all distances at once
    rigidity_matrix = kernel_function(pairwise_distances, eta, *kernel_params)
    # Ensure the diagonal is zero since we do not calculate rigidity for the same atom
    np.fill_diagonal(rigidity_matrix, 0)
    
    return rigidity_matrix


# Atomic Flexibility Index
def atomic_flexibility_index(rigidity_indices):
    flexibility_indices = 1.0 / np.maximum(rigidity_indices, np.finfo(float).eps)
    return flexibility_indices




# B-factor Prediction
def predict_b_factors(flexibility_indices, a, b):
    b_factors = a * flexibility_indices + b
    return b_factors



# Binding Induced Flexibility Reduction and Rigidity Strengthening --- DELTA B 
def calculate_delta_B(b_factors_complex, b_factors_protein):
    # Ensure we do not divide by zero
    b_factors_protein = np.maximum(b_factors_protein, np.finfo(float).eps)
    
    # Calculate ΔB using the formula provided
    delta_B = np.sum(np.abs(b_factors_complex - b_factors_protein) / b_factors_protein)
    
    return delta_B





# Rigidity Index Based Scoring Functions ------- RI SCORE 
def rigidity_index_scoring(protein_coords, ligand_coords, eta, kernel_function, kernel_params, cutoff):
    # Calculate pairwise distances between protein and ligand
    pairwise_distances = cdist(protein_coords, ligand_coords)

    # Apply the kernel function to the distances within the cutoff
    ri_matrix = kernel_function(pairwise_distances, eta, *kernel_params)

    # Apply the cutoff to the RI matrix
    ri_matrix[pairwise_distances > cutoff] = 0

    # Summing over all elements to get the rigidity index (RI)
    ri_score = np.sum(ri_matrix)
    return ri_score





def load_and_process_pdb(filepath):
    ppdb = PandasPdb().read_pdb(filepath)
    atom_df = ppdb.df['ATOM']
    coordinates = atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return coordinates

def prepare_features(coordinates, eta, function, params):
    rigidity_matrix = atomic_rigidity_index_matrix(coordinates, eta, function, params)
    flexibility_indices = atomic_flexibility_index(rigidity_matrix)
    return flexibility_indices.flatten()




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

# constants for B-factor prediction
a = 1.0
b = 0.0

# Predict B-factors
b_factors_exp = predict_b_factors(flexibility_indices_exp, a, b)
b_factors_lorentz = predict_b_factors(flexibility_indices_lorentz, a, b)

# Example CSV output (you may adjust as needed)
df_exp = pd.DataFrame(rigidity_matrix_exp)
df_lorentz = pd.DataFrame(rigidity_matrix_lorentz)

df_exp.to_csv('rigidity_indices_exponential.csv', index=False)
df_lorentz.to_csv('rigidity_indices_lorentz.csv', index=False)

print("Rigidity and flexibility indices have been saved to CSV files.")



############################################################################################################        

# Prepare features using exponential and Lorentz functions
features_exp = prepare_features(coordinates, eta, gen_exp_function, (kappa,))
features_lorentz = prepare_features(coordinates, eta, gen_lorentz_function, (nu,))

# Placeholder target data, should be replaced with actual data
target_data = np.random.random(features_exp.shape[0])

# Combine features from both kernel functions for a more robust model
features_combined = np.vstack((features_exp, features_lorentz)).T

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_combined, target_data, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting regressor
regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
regressor.fit(X_train, y_train)

# Predict on the testing set and calculate metrics
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Cross-validation to check for overfitting
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Average Score: {np.mean(scores)}")

# Feature importance analysis
feature_importance = regressor.feature_importances_
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Actual and Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()




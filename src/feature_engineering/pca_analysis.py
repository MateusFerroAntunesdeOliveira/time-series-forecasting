import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from utility.load_constants import reading_date_column, datetime_format
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
import utility.utils as utils

def load_data(file_path):
    return utils.read_csv_file_as_dataframe(file_path)

def apply_classical_pca(df, n_components=None):
    logger.debug(f"Applying PCA on DataFrame with shape: {df.shape}")

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Calculate the covariance matrix
    cov_mat = np.cov(df_scaled.T)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]
    logger.debug(f"Eigenvalues: {eig_vals}")
    logger.debug(f"Eigenvectors: {eig_vecs}")

    # Calculate the explained variance ratio
    total = sum(eig_vals)
    var_exp = [(i / total) for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)

    # Select the number of components
    if n_components:
        eig_vecs = eig_vecs[:, :n_components]

    # Project the data onto the principal components
    X_pca = df_scaled.dot(eig_vecs)
    transformed_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(eig_vecs.shape[1])])

    return transformed_df, eig_vals, cum_var_exp

def visualize_pca_components(df):
    logger.info("Visualizing PCA components")

    # Select numerical columns for PCA
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    data = df[numerical_columns]

    # Apply PCA
    transformed_data, eig_vals, cum_var_exp = apply_classical_pca(data)
    logger.debug(f"Transformed data shape: {transformed_data.shape}")

    # Create an elbow plot to visualize the explained variance ratio
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, marker='o')
    ax.set_title("Elbow Plot")
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    # Plot the eigenvalues to visualize the importance of each principal component
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(eig_vals) + 1), eig_vals, alpha=0.5, align='center', label='Individual explained variance')
    ax.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='Cumulative explained variance')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance')
    ax.set_title("Explained Variance for Each Principal Component")
    ax.legend(loc='best')
    plt.show()
    
def apply_pca_analysis():
    logger.info("Applying PCA Analysis")
    df = load_data(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    visualize_pca_components(df)
    logger.info("PCA Analysis Completed")

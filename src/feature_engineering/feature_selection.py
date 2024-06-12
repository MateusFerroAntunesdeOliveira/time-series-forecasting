import sys
import os
import pandas as pd

# Sklearn Imports
from sklearn.feature_selection import VarianceThreshold

# Custom Imports
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME, OUTPUT_SELECTED_FEATURES_FILENAME
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def feature_selection_based_on_correlation(df, correlation_threshold):
    """ Perform feature selection based on Pearson and Spearman correlation. """
    # Separate features and target - assuming the target is the last column
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Calculate Pearson & Spearman correlation - linear relationship & monotonic relationship
    pearson_corr = X.corrwith(y, method="pearson").abs()
    spearman_corr = X.corrwith(y, method="spearman").abs()
    # Combine results
    correlation_scores = pd.DataFrame({"Pearson": pearson_corr, "Spearman": spearman_corr})

    # Filter features based on correlation threshold
    pearson_selected_features = correlation_scores[correlation_scores["Pearson"] > correlation_threshold].index.tolist()
    spearman_selected_features = correlation_scores[correlation_scores["Spearman"] > correlation_threshold].index.tolist()

    selected_features = list(set(pearson_selected_features + spearman_selected_features))
    selected_dataframe = create_dataframe_with_selected_features(df, selected_features)
    return selected_features, selected_dataframe

def feature_selection_based_on_variance(df, variance_threshold):
    """ Perform feature selection based on variance threshold. """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    # Separate features and target - assuming the target is the last column
    X = numeric_df.iloc[:, :-1]

    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X)

    # Get support and selected feature names
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_feature_indices].tolist()
    selected_dataframe = create_dataframe_with_selected_features(df, selected_features)
    return selected_features, selected_dataframe

def create_dataframe_with_selected_features(df, selected_features):
    """ Create a new DataFrame with the selected features and the target column. """
    return df[selected_features + [df.columns[-1]]]

def apply_feature_selection():
    logger.info("Applying feature selection")

    # Read the filtered merged file
    df = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    logger.info(f"Original DataFrame shape: {df.shape}")

    # Perform feature selection based on correlation
    selected_features_correlation, corr_df = feature_selection_based_on_correlation(df, correlation_threshold=0.5)
    logger.info("")
    logger.info(f"Selected features based on correlation: {selected_features_correlation}")
    logger.debug(f"DataFrame shape based on correlation: {corr_df.shape}")
    logger.debug(f"DataFrame Features: {corr_df.columns.to_list()}")

    # Perform feature selection based on variance
    selected_features_variance, var_df = feature_selection_based_on_variance(df, variance_threshold=0.1)
    logger.info("")
    logger.info(f"Selected features based on variance: {selected_features_variance}")
    logger.debug(f"DataFrame shape based on variance: {var_df.shape}")
    logger.debug(f"DataFrame Features: {var_df.columns.to_list()}")

    # Combine selected features from both methods
    combined_features = list(set(selected_features_correlation + selected_features_variance))
    combined_df = create_dataframe_with_selected_features(df, combined_features)
    logger.info("")
    logger.debug(f"DataFrame shape based on combined features: {combined_df.shape}")
    
    # Save the combined DataFrame to a CSV file
    output_file_path = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_SELECTED_FEATURES_FILENAME)
    utils.save_csv_file(combined_df, output_file_path)
    logger.debug(f"Selected features saved to {output_file_path}")
    
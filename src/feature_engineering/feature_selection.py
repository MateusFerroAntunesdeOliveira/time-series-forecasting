import sys
import os
import pandas as pd

# Custom Imports
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def feature_selection_based_on_correlation(df, correlation_threshold):
    """
    Perform feature selection based on Pearson and Spearman correlation.
    
    :param df: DataFrame containing features and target.
    :param correlation_threshold: Threshold for selecting features based on correlation.
    :return: DataFrame with selected features.
    """
    # Separate features and target - assuming the target is the last column
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Calculate Pearson correlation - linear relationship
    pearson_corr = X.corrwith(y, method="pearson").abs()

    # Calculate Spearman correlation - monotonic relationship
    spearman_corr = X.corrwith(y, method="spearman").abs()

    # Combine results
    correlation_scores = pd.DataFrame({"Pearson": pearson_corr, "Spearman": spearman_corr})

    # Filter features based on correlation threshold
    pearson_selected_features = correlation_scores[correlation_scores["Pearson"] > correlation_threshold].index.tolist()
    spearman_selected_features = correlation_scores[correlation_scores["Spearman"] > correlation_threshold].index.tolist()

    logger.info("Features selected using Pearson correlation: %s", pearson_selected_features)
    logger.info("Features selected using Spearman correlation: %s", spearman_selected_features)

    # Create DataFrame with selected features
    selected_features = list(set(pearson_selected_features + spearman_selected_features))
    selected_df = df[selected_features + [df.columns[-1]]]  # Add target column to the selected features DataFrame

    return selected_df

def apply_feature_selection():
    logger.info("Applying feature selection")
    
    # Read the filtered merged file
    df = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    logger.info(f"Original DataFrame shape: {df.shape}")
    
    # Perform feature selection based on correlation
    selected_df = feature_selection_based_on_correlation(df, correlation_threshold=0.5)
    logger.info(f"Selected DataFrame shape: {selected_df.shape}")
    logger.info(f"Selected features based on correlation: {selected_df.columns.tolist()}")

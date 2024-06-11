import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps

from data_processing.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
from data_processing.load_constants import target_column
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def read_csv_file_as_dataframe(file_path):
    logger.debug(f"Reading file {file_path}")
    return pd.read_csv(file_path, sep=";", encoding="utf-8")

def filter_zero_values(df, column_name):
    logger.debug("Filtering zero values")
    return df[df[column_name] != 0]

def apply_statistics(df):
    logger.debug("Applying statistics")
    return df.describe()
   
def plot_symmetry(df, column_name):
    logger.info("Plotting symmetry")
    mean, median, mode = evaluate_mean_median_mode(df, column_name)

    plt.figure(figsize=(10, 8))
    # Kernel Density Estimation (KDE) is a way to estimate the probability density function of a continuous random variable.
    sns.histplot(df[column_name], bins=30, kde=True, color="blue", stat="density")
    plt.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.2f}")
    plt.axvline(median, color="green", linestyle="-", linewidth=2, label=f"Median: {median:.2f}")
    plt.axvline(mode, color="purple", linestyle="-", linewidth=2, label=f"Mode: {mode:.2f}")
    plt.title(f"Distribution of {column_name} with Measures of Central Tendency")
    plt.xlabel(column_name)
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    logger.info("Symmetry plot created")

# TODO:
# Create more functions to plot other types of visualizations (e.g., boxplot, scatterplot, heatmap, histogram, etc.)

def evaluate_mean_median_mode(df, column_name):
    logger.debug("Evaluating mean, median, and mode")
    return df[column_name].mean(), df[column_name].median(), df[column_name].mode()[0]

def remove_constant_columns(df):
    logger.debug("Removing constant columns")
    return df.loc[:, df.apply(pd.Series.nunique) != 1]

def get_pearson_correlation(df, method="pearson"):
    logger.debug("Calculating Pearson correlation")
    return df.corr(method=method)

def get_predictive_power_score_correlation(df):
    logger.debug("Calculating Predictive Power Score (PPS) correlation")
    # PPS calculation for each feature with target variable
    pps_matrix = pps.matrix(df)

    # Drop rows where the feature is the target variable itself & Extract numerical values from the PPS matrix
    pps_matrix = pps_matrix[pps_matrix["x"] != pps_matrix["y"]]
    pps_matrix_values = pps_matrix.pivot(index="x", columns="y", values="ppscore")
    return pps_matrix_values

def plot_correlation_matrix(corr, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()

def apply_measures():
    logger.info("Applying measures")
    merged_csv_file = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME)

    # Describe the dataset & Plot symmetry with original dataset
    df_all = read_csv_file_as_dataframe(merged_csv_file)
    summary_stats = apply_statistics(df_all)
    logger.info(f"Summary Statistics with original dataset:\n{summary_stats}")
    plot_symmetry(df_all, target_column)

    # Filter zero values before applying statistics - In this case, the zero values are not good for the analysis
    # Describe the dataset & Plot symmetry with filtered dataset
    df_filtered = filter_zero_values(df_all, target_column)
    summary_stats = apply_statistics(df_filtered)
    logger.info(f"Summary Statistics with filtered dataset:\n{summary_stats}")
    plot_symmetry(df_filtered, target_column)

    # Plot correlation heatmap with filtered dataset
    logger.info("Plotting correlation heatmap")
    df = remove_constant_columns(df)
    correlation = get_pearson_correlation(df)
    pps_correlation = get_predictive_power_score_correlation(df)
    plot_correlation_matrix(correlation, "Pearson Correlation matrix")
    plot_correlation_matrix(pps_correlation, "Predictive Power Score (PPS) matrix")

    logger.info("Measures applied")

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
from data_processing.load_constants import target_column
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def read_csv_file_as_dataframe(file_path):
    logger.debug(f"Reading file {file_path}")
    return pd.read_csv(file_path, sep=';', encoding='utf-8')

def filter_zero_values(df, column_name):
    logger.info("Filtering zero values")
    return df[df[column_name] != 0]

def apply_statistics(df):
    logger.info("Applying statistics")
    return df.describe()
   
def plot_symmetry(df, column_name):
    logger.info("Plotting symmetry")
    mean, median, mode = evaluate_mean_median_mode(df, column_name)

    plt.figure(figsize=(10, 6))
    # Kernel Density Estimation (KDE) is a way to estimate the probability density function of a continuous random variable.
    sns.histplot(df[column_name], bins=30, kde=True, color='blue', stat='density')
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(mode, color='purple', linestyle='-', linewidth=2, label=f'Mode: {mode:.2f}')
    plt.title(f'Distribution of {column_name} with Measures of Central Tendency')
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    logger.info("Symmetry plot created")

# TODO:
# Create more functions to plot other types of visualizations (e.g., boxplot, scatterplot, heatmap, histogram, etc.)

def evaluate_mean_median_mode(df, column_name):
    logger.info("Evaluating mean, median, and mode")
    return df[column_name].mean(), df[column_name].median(), df[column_name].mode()[0]

def remove_constant_columns(df):
    logger.info("Removing constant columns")
    return df.loc[:, df.apply(pd.Series.nunique) != 1]

def get_correlation(df):
    logger.info("Calculating correlation")
    return df.corr()

def plot_correlation_matrix(corr):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()

def get_correlation_heatmap(df):
    logger.info("Plotting correlation heatmap")
    df = remove_constant_columns(df)
    correlation = get_correlation(df)
    plot_correlation_matrix(correlation)
    logger.info("Correlation heatmap created")

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
    get_correlation_heatmap(df_filtered)

    logger.info("Measures applied")

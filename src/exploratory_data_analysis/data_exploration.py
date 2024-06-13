import pandas as pd

from ydata_profiling import ProfileReport

from utility.load_constants import reading_date_column
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME, OUTPUT_FILTERED_MERGED_FILENAME, OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME
import utility.utils as utils

def perform_eda():
    utils.create_and_check_directory(OUTPUT_EDA_REPORT_PATH)

    logger.info("Performing Exploratory Data Analysis (EDA)")

    # Read the merged file
    df_all = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    logger.info(f"Loaded Merged DataFrame: {OUTPUT_MERGED_FILENAME} with shape: {df_all.shape}")

    # Read the filtered merged file
    df_filtered = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME))
    logger.info(f"Loaded Filtered Merged DataFrame: {OUTPUT_FILTERED_MERGED_FILENAME} with shape: {df_filtered.shape}")

    # Read the filtered merged file for first month
    df_filtered_first_month = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME))
    df_filtered_first_month = df_filtered_first_month.head(720)

    # Convert columns to appropriate data types
    df_all[reading_date_column] = pd.to_datetime(df_all[reading_date_column], format='%Y-%m-%d %H:%M:%S')
    df_filtered[reading_date_column] = pd.to_datetime(df_filtered[reading_date_column], format='%Y-%m-%d %H:%M:%S')
    df_filtered_first_month[reading_date_column] = pd.to_datetime(df_filtered_first_month[reading_date_column], format='%Y-%m-%d %H:%M:%S')

    # Perfome EDA using ydata_profiling
    logger.info("Performing EDA using ydata_profiling")
    profile = ProfileReport(df_all, title="EDA Report")
    profile.to_file(utils.join_file_path(OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME))
    logger.info(f"EDA Report saved to: {OUTPUT_EDA_REPORT_FILENAME}")

    # Perfome EDA using ydata_profiling with time series mode
    logger.info("Performing EDA using ydata_profiling with time series mode")
    profile = ProfileReport(df_filtered, title="EDA Report with TS mode", tsmode=True, sortby=reading_date_column)
    profile.to_file(utils.join_file_path(OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME.replace(".html", "_ts.html")))
    logger.info(f"EDA Report with TS mode saved to: {OUTPUT_EDA_REPORT_FILENAME.replace('.html', '_ts.html')}")

    # Perfome EDA using ydata_profiling with time series mode with only 1 month range
    logger.info("Performing EDA using ydata_profiling with time series mode with only 1 month range")
    profile = ProfileReport(df_filtered_first_month, title="EDA Report with TS mode for 1 month range", tsmode=True, sortby=reading_date_column)
    profile.to_file(utils.join_file_path(OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME.replace(".html", "_ts_1m.html")))
    logger.info(f"EDA Report with TS mode for 1 month range saved to: {OUTPUT_EDA_REPORT_FILENAME.replace('.html', '_ts_1m.html')}")

    logger.info("Exploratory Data Analysis (EDA) completed")

def explore_data():
    logger.info("Exploring data")
    perform_eda()

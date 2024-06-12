from ydata_profiling import ProfileReport

from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME, OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME
import utility.utils as utils

def perform_eda():
    logger.info("Performing Exploratory Data Analysis (EDA)")

    # Read the filtered merged file
    df = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    logger.info(f"DataFrame shape: {df.shape}")

    # Perfome EDA using ydata_profiling
    profile = ProfileReport(df, title="EDA Report")
    profile.to_file(utils.join_file_path(OUTPUT_EDA_REPORT_PATH, OUTPUT_EDA_REPORT_FILENAME))

def explore_data():
    logger.info("Exploring data")
    perform_eda()

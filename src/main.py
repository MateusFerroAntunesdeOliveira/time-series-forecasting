import data_processing.process_data as process_data
import data_processing.data_cleaning as data_cleaning
import data_processing.data_visualization as data_visualization
import feature_engineering.feature_prediction as feature_prediction
import feature_engineering.feature_selection as feature_selection
import exploratory_data_analysis.data_exploration as data_exploration

def main():
    process_data.process_files()
    data_cleaning.clean_data()
    data_visualization.apply_measures()
    feature_prediction.apply_feature_prediction_classification()
    feature_selection.apply_feature_selection()
    data_exploration.explore_data()

if __name__ == "__main__":
    main()

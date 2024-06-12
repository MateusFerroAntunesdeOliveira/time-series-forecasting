import data_processing.process_and_merge_csv as process_and_merge_csv
import data_processing.data_visualization as data_visualization
import data_processing.data_cleaning as data_cleaning
import feature_engineering.predictive_features as predictive_features
import feature_engineering.feature_selection as feature_selection

def main():
    # process_and_merge_csv.process_files()
    # data_visualization.apply_measures()
    # data_cleaning.clean_data()
    # predictive_features.apply_classification()
    feature_selection.apply_feature_selection()

if __name__ == "__main__":
    main()

import data_processing.process_data as process_data
import data_processing.data_cleaning as data_cleaning
import data_processing.data_visualization as data_visualization
import exploratory_data_analysis.data_exploration as data_exploration
import feature_engineering.feature_selection as feature_selection
import feature_engineering.pca_analysis as pca_analysis
import feature_engineering.autoencoder_analysis as autoencoder_analysis
import feature_engineering.feature_prediction as feature_prediction
import forecasting.lstm_time_series as lstm_time_series
import forecasting.gru_time_series as gru_time_series
import forecasting.arima_time_series as arima_time_series

def main():
    process_data.process_files()
    data_cleaning.clean_data()
    data_visualization.apply_measures()
    data_exploration.explore_data()
    feature_selection.apply_feature_selection()
    pca_analysis.apply_pca_analysis()
    autoencoder_analysis.apply_autoencoder_analysis()
    feature_prediction.apply_feature_prediction_classification()
    lstm_time_series.apply_lstm_forecasting(1)
    lstm_time_series.apply_lstm_forecasting(12)
    lstm_time_series.apply_lstm_forecasting(24)
    lstm_time_series.apply_lstm_forecasting(48)
    gru_time_series.apply_gru_forecasting(1)
    gru_time_series.apply_gru_forecasting(12)
    gru_time_series.apply_gru_forecasting(24)
    gru_time_series.apply_gru_forecasting(48)
    arima_time_series.apply_arima_forecasting(1)
    arima_time_series.apply_arima_forecasting(12)
    arima_time_series.apply_arima_forecasting(24)
    arima_time_series.apply_arima_forecasting(48)

if __name__ == "__main__":
    main()

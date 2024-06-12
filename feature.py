import load_data
import numpy as np
from scipy.stats import linregress

numerical_columns = ['Lin-Acc_X (m/s^2)', 'Lin-Acc_Y (m/s^2)', 'Lin-Acc_Z (m/s^2)', 'Location_Latitude (°)', 'Location_Longitude (°)', 'Location_Height (m)', 'Location_Horizontal Accuracy (m)', 'Location_Vertical Accuracy (°)', 'Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)', 'Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)']
def add_numerical_features(data_resampled, numerical_list=[], window_size=3):
    """
    Adds numerical features (Min, Max, STD, Mean, Slope) to the data resampled.

    Args:
        data_resampled: list of df
        numerical_list: list of numerical columns
        window_size: window size for calculation
    Returns:
        data_numerical_featured: list of df with numerical features
    """
    data_numerical_featured = []

    for df in data_resampled:
        for col in numerical_list:
            # Calculate rolling features
            df[f'{col}_Min'] = df[col].rolling(window=window_size).min()
            df[f'{col}_Max'] = df[col].rolling(window=window_size).max()
            df[f'{col}_STD'] = df[col].rolling(window=window_size).std()
            df[f'{col}_Mean'] = df[col].rolling(window=window_size).mean()

            # Calculate slope (trend) using a rolling window
            def calculate_slope(series):
                if len(series) < window_size:
                    return 0
                y = series.values
                x = range(len(y))
                slope, _, _, _, _ = linregress(x, y)
                return slope

            df[f'{col}_Slope'] = df[col].rolling(window=window_size).apply(calculate_slope, raw=False)

        data_numerical_featured.append(df)

    return data_numerical_featured


if __name__ == '__main__':
    _, data_resampled = load_data.process_data()
    data_resampled = data_resampled['100ms']
    data_numerical_featured = add_numerical_features(data_resampled, numerical_columns)
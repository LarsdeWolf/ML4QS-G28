import numpy as np
import pandas as pd
from load_data import *
from scipy.stats import linregress


label_to_id = {'walk': 0, 'run': 1, 'bike': 2, 'car': 3, 'train': 4}


def extract_features(data: list, sensors: list, window: int=10):
    """
    Extracts features for the given sensors using a sliding window
    Args:
        data: list of pd.dataframes
        sensors: list of sensor strings to include for feature extraction
        window: window size for feature calculation

    Returns:
        X: numpy feature array  (N, n_features)
        y: numpy label array    (N,)

    """
    def features(window_data):
        """
        TODO: implement features
        calculate features for every column of window_data (only sensory measurements)

        Args:
            window_data: window_sized df

        Returns:
            np.array
        """
        features = []
        numerical_columns = ['Lin-Acc_X (m/s^2)', 'Lin-Acc_Y (m/s^2)', 'Lin-Acc_Z (m/s^2)', 'Location_Latitude (°)',
                             'Location_Longitude (°)', 'Location_Height (m)', 'Location_Velocity (m/s)', 'Location_Direction (°)',
                             'Location_Horizontal Accuracy (m)','Location_Vertical Accuracy (°)', 'Accelerometer_X (m/s^2)',
                             'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)', 'Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)',
                             'Gyroscope_Z (rad/s)']

        for col in numerical_columns:
            # Calculate rolling features
            features.append(window_data[col].min())
            features.append(window_data[col].max())
            features.append(window_data[col].mean())
            features.append(window_data[col].std())
            # Calculate slope using linear regression
            y = window_data[col].values
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            features.append(slope)

        return np.array(features)


    X, y = [], []
    # Columns to exclude
    drop_colls = {'Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location'} - set(sensors)
    for df in data:
        df = df.drop([col for col in df.columns if any(sensor in col for sensor in drop_colls)], axis=1)
        # Taking groups of data of window size
        for i in range(len(df) - window):
            values = df.iloc[i: i + window]  # Taking values over window
            X.append(features(values))
            y.append(label_to_id[values[['walk', 'run', 'bike', 'car', 'train']].iloc[0].idxmax()])

    return np.array(X), np.array(y)


if __name__ == '__main__':
    _, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = data_resampled['100ms']
    X, y = extract_features(data, sensors)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Features extracted")

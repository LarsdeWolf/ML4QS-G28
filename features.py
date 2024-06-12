import numpy as np
import pandas as pd
from load_data import *


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
        return np.array
        """
        pass

    X, y = np.array([]), np.array([])
    # Columns to exclude
    drop_colls = {'Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location'} - set(sensors)
    for df in data:
        df = df.drop([col for col in df.columns if any(sensor in col for sensor in drop_colls)], axis=1)
        # Taking groups of data of window size
        for i in range(len(df) - window):
            values = df.iloc[i: i + window]  # Taking values over window
            X.append(features(values))
            y.append(label_to_id[values[['walk', 'run', 'bike', 'car', 'train']].iloc[0].idxmax(axis=1)])

    return X, y


if __name__ == '__main__':
    _, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = data_resampled['100ms']


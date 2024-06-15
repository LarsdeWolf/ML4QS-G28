import pandas as pd
import numpy as np
from load_data import *
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

def clean_data(data_list, contamination=0.1, n_neighbors=20):
    """
    Cleans the input data by detecting and handling outliers and missing values.

    Args:
        data: list of dfs
        contamination: The proportion of outliers in the data
        n_neighbors: Number of neighbors to use for LOF. Default is 20.

    Returns:
        pd.DataFrame: The cleaned list of dfs
    """
    # Identify numeric columns, excluding 'timestamp' if present
    cleaned_data_list = []

    for data in data_list:
        # Identify numeric columns, excluding 'timestamp' if present
        numeric_cols = ['Lin-Acc_X (m/s^2)', 'Lin-Acc_Y (m/s^2)', 'Lin-Acc_Z (m/s^2)',
                        'Location_Latitude (°)', 'Location_Longitude (°)', 'Location_Height (m)',
                        'Location_Velocity (m/s)', 'Location_Direction (°)', 'Location_Horizontal Accuracy (m)',
                        'Location_Vertical Accuracy (°)', 'Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)',
                        'Accelerometer_Z (m/s^2)', 'Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)']

        # Initial imputation of missing values with mean
        imputer = SimpleImputer(strategy='mean')
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

        # Apply LOF for outlier detection on each numeric column
        for col in numeric_cols:
            # Detect outliers using LOF
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            is_outlier = lof.fit_predict(data[[col]])
            data.loc[is_outlier == -1, col] = np.nan  # Mark outliers as NaN

        # Final interpolation of missing values
        data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit_direction='both')

        cleaned_data_list.append(data)

    return cleaned_data_list


def plot_data(original, cleaned, title):
    """Plot original and cleaned data."""
    if original is None or cleaned is None:
        return

    plt.figure(figsize=(12, 5))
    if 'timestamp' in original.columns:
        time_col = original['timestamp']
    else:
        time_col = range(len(original))

    # numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    # numeric_cols = ['Lin-Acc_X (m/s^2)', 'Lin-Acc_Y (m/s^2)', 'Lin-Acc_Z (m/s^2)',
    #                 'Location_Latitude (°)', 'Location_Longitude (°)',
    #                 'Location_Height (m)', 'Location_Velocity (m/s)',
    #                 'Location_Direction (°)', 'Location_Horizontal Accuracy (m)',
    #                 'Location_Vertical Accuracy (°)', 'Accelerometer_X (m/s^2)',
    #                 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)',
    #                 'Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)']
    numeric_cols = ['Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)','Gyroscope_X (rad/s)']

    cols_to_plot = [col for col in numeric_cols if col != 'timestamp']
    num_plots = len(cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        plt.subplot(1, num_plots, i + 1)
        plt.plot(time_col, original[col], label='Original', alpha=0.5)
        plt.plot(time_col, cleaned[col], label='Cleaned', alpha=0.5)
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    _, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = data_resampled['100ms']
    cleaned_data = clean_data(data, contamination=0.1, n_neighbors=20)
    plot_data(data[0], cleaned_data[0], "1")

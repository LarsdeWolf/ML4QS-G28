import numpy as np
from load_data import *
from cleaning import *
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.impute import SimpleImputer

label_to_id = {'walk': 0, 'run': 1, 'bike': 2, 'car': 3, 'train': 4}


def extract_features(data: list, sensors: list, window: int = 10, overlap: float = 0.5, multi_processing: bool = False,
                     close: bool = False):
    """
    Extracts features for the given sensors using a sliding window
    Args:
        data: list of pd.dataframes
        sensors: list of sensor strings to include for feature extraction
        window: window size for feature calculation
        overlap: amount of overlap for windows
        multi_processing: use multiprocessing (install pathos)
        close: close pool (only if you know this is the last extract_features() call (i.e. single use)

    Returns:
        X: numpy feature array  (N, n_features)
        y: numpy label array    (N,)

    """

    def features(window_data):
        """
        TODO: implement features
        calculate features for every column of window_data (only sensory measurements)

        Args:
            window_data: window_sized np array (1, window_size, n_sensors)

        Returns:
            np.array
        """
        min_values = np.min(window_data, axis=1)[0]
        max_values = np.max(window_data, axis=1)[0]
        mean_values = np.mean(window_data, axis=1)[0]
        std_values = np.std(window_data, axis=1)[0]
        x = np.arange(window_data.shape[1])
        slopes = np.array([np.polyfit(x, window_data[:, :, i][0], 1)[0] for i in range(window_data.shape[2])])
        # Compute Fourier Transform for each sensor data in the window (Book 2.2.2)
        fourier_transform = np.abs(np.fft.fft(window_data[0, :, :], axis=0))
        fourier_max = fourier_transform.max(axis=0)
        fourier_mean = fourier_transform.mean(axis=0)

        features = np.concatenate([min_values, max_values, mean_values, std_values, slopes, fourier_max, fourier_mean])
        return features

    if multi_processing:
        p = Pool()

    X, y = [], []
    drop_colls = {'Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location'} - set(sensors)
    label_colls = {'walk', 'run', 'bike', 'car', 'train'}
    drop_colls.update(label_colls)
    drop_colls.update({'id', 'Time (ns)'})

    for df in data:
        label = label_to_id[df[list(label_colls)].iloc[0].idxmax()]  # Labels are assumed to be the same throughout df
        df = df.drop(columns=[col for col in df.columns if any(sensor in col for sensor in drop_colls)])

        df_np = df.to_numpy()

        # Check for NaN values to avoid error in slope
        if np.isnan(df_np).any():
            # print("Data contains NaN values. Imputing missing values.")
            imputer = SimpleImputer(strategy='mean')
            df_np = imputer.fit_transform(df_np)

        stride = max(1, int(window * (1 - overlap)))
        windowed_data = np.lib.stride_tricks.sliding_window_view(df_np, (window, df_np.shape[1]))[::stride, :, :]
        if multi_processing:
            result = p.map(features, windowed_data)
        else:
            result = [features(wd) for wd in windowed_data]
        X.extend(result)
        y.extend([label] * len(windowed_data))

    if multi_processing and close:
        print("Closed Pool!")
        p.close()
        p.join()

    return np.array(X, dtype=object), np.array(y)


if __name__ == '__main__':
    _, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = data_resampled['100ms']
    cleaned_data = clean_data(data)
    X, y = extract_features(cleaned_data, sensors, multi_processing=True, close=True)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Features extracted")

import numpy as np
from features import *


label_to_id = {'walk': 0, 'run': 1, 'bike': 2, 'car': 3, 'train': 4}


def train_test_split_measurementlevel(X, y, train_size=0.75, test_size=0.15, dev_size=0.10):
    """
    Shuffles and splits the datapoints into training, testing and validation sets
    """
    perm = np.random.permutation(len(X))
    train_split, test_split, dev_split = int(len(X) * train_size), int(len(X) * test_size), int(len(X) * dev_size)

    X_train, y_train = X[perm[:train_split]], y[perm[:train_split]]
    X_test, y_test = X[perm[train_split:train_split + test_split]], y[perm[train_split:train_split + test_split]]
    X_dev, y_dev = X[perm[train_split + test_split:len(X)]], y[perm[train_split + test_split:len(X)]]

    return X_train, y_train, X_test, y_test, X_dev, y_dev


def train_test_split_activitylevel(data):
    """
    Splits the data list of pandas frames, such that 1 measurement is taken for the test/dev set, rest
    for training.
    Args:
        data:

    Returns: train, test, dev lists of dataframes

    """
    label_dict = {label: [df for df in data if df[label].iloc[0] == 1] for label in
                  ['walk', 'run', 'bike', 'car', 'train']}
    data_train = [df for dfs in label_dict.values() for df in dfs[1:]]  # All but first
    data_test_dev = [df for dfs in label_dict.values() for df in dfs[:1]]    # First
    data_test = data_test_dev[:len(data_test_dev) // 2]
    data_dev = data_test_dev[len(data_test_dev) // 2:]

    return data_train, data_test, data_dev

def np_from_df(data, step_size):
    """
    Converts a list of dataframes feature and label arrays
    Uses step_size to create sequences of data
    Assumes all elements in the dataframe have the same label
    Args:
        data: list of dataframes
        step_size: size of sequence

    Returns:
        X: numpy array of features of shape (n_samples, step_size, n_features)
        y: numpy array of labels
    """
    X, y = [], []
    for df in data:
        if len(df) < 1:
            continue
        label = label_to_id[df[['walk', 'run', 'bike', 'car', 'train']].iloc[0].idxmax()]
        df = df.drop(['walk', 'run', 'bike', 'car', 'train', 'Time (ns)', 'id'], axis=1)
        for row in range(len(df) - step_size):
            features = df.iloc[row: row + step_size, df.columns != 'label'].values
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

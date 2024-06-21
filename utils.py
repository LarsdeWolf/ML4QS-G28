import numpy as np
import pickle
import torch
from itertools import product
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pathos.multiprocessing import ProcessingPool as Pool
from features import extract_features

label_to_id = {'walk': 0, 'run': 1, 'bike': 2, 'car': 3, 'train': 4}


class DataLoader(torch.utils.data.DataLoader):
    # Faster Dataloader
    # https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


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
    data_test_dev = [df for dfs in label_dict.values() for df in dfs[:1]]  # First
    data_test = data_test_dev[:len(data_test_dev) // 2]
    data_dev = data_test_dev[len(data_test_dev) // 2:]

    return data_train, data_test, data_dev


def np_from_df(data, sensors, step_size, multip=False, close=False):
    """
    Converts a list of dataframes to feature and label arrays. (using the raw data)
    Uses step_size to create sequences of data
    Assumes all elements in the dataframe have the same label
    Args:
        data: list of dataframes
        sensors: sensors to include
        step_size: size of sequence
        close: close the multiprocessing pool (last extraction of df in current session)
        multip: use multiprocessing

    Returns:
        X: numpy array of features of shape (n_samples, step_size, n_features)
        y: numpy array of labels
    """

    def return_features(i):
        features = df.iloc[i: i + step_size, df.columns != 'label'].values
        return features

    X, y = [], []
    drop_colls = {'Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location'} - set(sensors)
    label_colls = {'walk', 'run', 'bike', 'car', 'train'}
    drop_colls.update(label_colls)
    drop_colls.update({'id', 'Time (ns)'})

    if multip:
        p = Pool()

    for df in data:
        if len(df) < 1:
            continue
        label = label_to_id[df[list(label_colls)].iloc[0].idxmax()]
        df = df.drop(columns=[col for col in df.columns if any(sensor in col for sensor in drop_colls)])
        if multip:
            result = p.map(return_features, np.arange(len(df) - step_size))
        else:
            result = [return_features(i) for i in range(len(df) - step_size)]
        X.extend(result)
        y.extend([label] * (len(df) - step_size))
    if multip and close:
        print("Closed Pool!")
        p.close()
        p.join()
    return np.array(X), np.array(y)


def get_data(data, sensors, dataset_level, model, window_stepsize, multi_p, close):
    """
    automatically collect data for given model, data, stepsize, etc
    Args:
        data: List of pd.Dataframes
        sensors: List of sensors to use
        dataset_level: measurement or activity based train/test/dev splitting
        model: LSTM/KNN/DT
        window_stepsize: KNN/DT: aggregation window size, LSTM: sequence/step size (#points in a datapoint)
        multi_p: use multiprocessing
        close: close multip. pool (last run)

    Returns:
        data: [X_train, y_train, X_test, y_test, X_dev, y_dev]
    """
    if model == 'LSTM':
        if dataset_level == 'measurement':
            X, y = np_from_df(list(df.dropna() for df in data), sensors, window_stepsize, multi_p, close)
            data = train_test_split_measurementlevel(X, y)
        else:
            data_train, data_test, data_dev = train_test_split_activitylevel(data)
            data = [
                *np_from_df(list(df.dropna() for df in data_train), sensors, window_stepsize, multi_p),  # Train
                *np_from_df(list(df.dropna() for df in data_test), sensors, window_stepsize, multi_p),  # Test
                *np_from_df(list(df.dropna() for df in data_dev), sensors, window_stepsize, multi_p, close)  # Dev
            ]
    else:
        if dataset_level == 'measurement':
            X, y = extract_features(data, sensors, multi_processing=multi_p, close=close)
            data = train_test_split_measurementlevel(X, y)
        else:
            data_train, data_test, data_dev = train_test_split_activitylevel(data)
            data = [
                *extract_features(data_train, sensors, window=window_stepsize, multi_processing=multi_p),  # Train
                *extract_features(data_test, sensors, window=window_stepsize, multi_processing=multi_p),  # Test
                *extract_features(data_dev, sensors, window=window_stepsize, multi_processing=multi_p, close=close)
                # Dev
            ]
    return data


def tuneHyperParameters(models, sensors, dataset_level, data, save=False):
    """
    Tunes the hyperparameters for each model using GridSearchCV and K-Fold validation
    Uses 10 window/sequence length
    Args:
        models: List of models
        data: list of data [X_train, y_train, X_dev, ... ,]
    Returns:
        model_params: dict with models as keys and dicts with optimal parameters as values
    """
    from train_LSTM import train
    kf = KFold(n_splits=5, shuffle=True)
    grid_values = {'DT': {'max_depth': [None, 5, 10, 20, 50],
                          'min_samples_split': [0.5, 4, 8, 16],
                          'min_samples_leaf': [1, 2, 4, 8]},
                   'KNN': {'leaf_size': [10, 20, 30, 40],
                           'weights': ['uniform', 'distance'],
                           'metric': ['euclidean', 'manhattan']},
                   'LSTM': {'hidden_size': [50, 100, 200, 400],
                            'drop_out': [0.3, 0.5, 0.8],
                            'lr': [1e-1, 1e-2, 1e-3]},
                   }
    model_params = {}
    close = False
    for model in models:
        # Tune parameters for LSTM model
        params = grid_values[model]
        if model == models[-1]:
            close = True
        data_model = get_data(data, sensors, dataset_level, model, 10, True, close)
        if model == 'LSTM':
            best_score = 0
            best_params = None

            combinations = list(product(*params.values()))
            for combination in combinations:
                params = dict(zip(params.keys(), combination))
                fold_scores = []
                for train_index, val_index in kf.split(data_model[0]):
                    # Split the data into training and validation sets
                    X_train, X_val = data_model[0][train_index], data_model[0][val_index]
                    y_train, y_val = data_model[1][train_index], data_model[1][val_index]
                    trained_model, acc_dev, _ = train((X_train, y_train, X_val, y_val, X_val, y_val), epochs=5,
                                                      hidden_size=params['hidden_size'], dropout=params['drop_out'],
                                                      lr=params['lr'])
                    fold_scores.append(acc_dev[0])

                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            model_params[model] = best_params

        else:
            if model == 'DT':
                classifier = DecisionTreeClassifier()
            else:
                classifier = KNeighborsClassifier(n_neighbors=5)
            grid_search = GridSearchCV(classifier, params, cv=kf, scoring='accuracy')
            grid_search.fit(data_model[0], data_model[1])  # Train & Test data
            best_params = grid_search.best_params_
        print(f"Best parameters for {model}: {best_params}")
        model_params[model] = best_params

    # save dict of model params to file
    if save:
        with open(f'Results/paramsForModels_{"_".join(models)}.pkl', 'wb') as f:
            pickle.dump(model_params, f)

    return model_params

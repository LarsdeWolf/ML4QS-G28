from train_DT import train as trainDT
from train_KNN import train as trainKNN
from train_LSTM import train as trainLSTM
from utils import *


def experiments(models, data, granularities, windows, sensors, data_level, repeats=3):
        for gran in granularities:
            data_gran = data[gran]
            for window in windows:
                dataset_saved = None
                for model in models:
                    if model in ['KNN', 'DT']:  # Dataset for KNN & DT are the same and can be re-used
                        if dataset_saved is None:  # If not saved, we get data and save
                            dataset = get_data(data_gran, sensors, data_level, model, window, True, False)
                            dataset_saved = dataset
                        else:  # Using saved dataset
                            dataset = dataset_saved
                    else:  # LSTM get data
                        dataset = get_data(data_gran, sensors, data_level, model, window, True, False)
                    for repeat in range(repeats):
                        print(f"Training {model} model on dataset with {gran} granularity and {window} window size ")
                        print(f'Repeat {repeat}')
                        experiment(model, dataset)
                        print()


def experiment(model, data):
    if model == 'LSTM':
        model, _ = trainLSTM(data, output=False, epochs=10)  # Suppressing train printing output in order to show summarized output (TODO)
    if model == 'KNN':
        model = trainKNN(data, output=False, epochs=10)
    if model == 'DT':
        model = trainDT(data, output=False, epochs=10)

if __name__ == '__main__':
    _, data_resampled = process_data()
    granularities = ['100ms', '500ms', '1s']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data_level = 'measurement'
    models = ['KNN', 'LSTM', 'DT']
    windows = [5, 10, 50]

    experiments(models, data_resampled, granularities, windows, sensors, data_level)

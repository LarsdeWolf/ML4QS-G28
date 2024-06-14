from train_DT import train as trainDT
from train_KNN import train as trainKNN
from train_LSTM import train as trainLSTM
from utils import *


def experiments(models, data, granularities, windows, sensors, data_level, repeats=3):
    for model in models:
        for gran in granularities:
            data_gran = data[gran]
            for window in windows:
                dataset = get_data(data_gran, sensors, data_level, model, window)
                for repeat in range(repeats):
                    print(f"Training {model} model on dataset with {gran} granularity and {window} window size ")
                    print(f'Repeat {repeat}')
                    experiment(model, dataset)


def experiment(model, data):
    if model == 'LSTM':
        trainLSTM(data, epochs=10)
    if model == 'KNN':
        trainKNN(data, epochs=10)
    else:
        trainDT(data, epochs=10)

if __name__ == '__main__':
    _, data_resampled = process_data()
    granularities = ['100ms', '500ms', '1s']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data_level = 'measurement'
    models = ['LSTM', 'KNN', 'DT']
    windows = [5, 10, 50, 100]

    experiments(models, data_resampled, granularities, windows, sensors, data_level)

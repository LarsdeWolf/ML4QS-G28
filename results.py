from train_DT import train as trainDT
from train_KNN import train as trainKNN
from train_LSTM import train as trainLSTM
from utils import get_data, tuneHyperParameters
from load_data import process_data
from cleaning import *
import os
import numpy as np
import pickle
import torch
import sys


def save_metrics(metrics, path):
    with open(path, 'wb') as f:
        pickle.dump(metrics, f)


def save_model(model, path):
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def experiments(models, params, data, granularities, windows, sensors, data_level, repeats=3):
    for gran in granularities:
        data_gran = data[gran]
        data_gran = clean_data(data_gran)
        for window in windows:
            dataset_saved = None
            best_accuracy = 0
            best_model = None
            best_metrics = None
            best_m = None
            for model in models:
                parameters = params[model]
                if model in ['KNN', 'DT']:  # Dataset for KNN & DT are the same and can be re-used
                    if dataset_saved is None:  # If not saved, we get data and save
                        dataset = get_data(data_gran, sensors, data_level, model, window, True, False)
                        dataset_saved = dataset
                    else:  # Using saved dataset
                        dataset = dataset_saved
                else:  # LSTM get data
                    dataset = get_data(data_gran, sensors, data_level, model, window, True, False)

                all_metrics = []  # Record each repeat metrics
                for repeat in range(repeats):
                    print(f"Training {model} model on dataset with {gran} granularity and {window} window size ")
                    print(f'Repeat {repeat}')
                    m, metrics = experiment(model, parameters, dataset)
                    all_metrics.append(metrics)

                    # Save the results of each run
                    run_dir = f"results/{gran}_{window}_{model}_repeat{repeat}"
                    os.makedirs(run_dir, exist_ok=True)
                    save_metrics(metrics, os.path.join(run_dir, 'metrics.pkl'))
                    save_model(m, os.path.join(run_dir, 'model.pkl'))
                    # print(f"Results saved in {run_dir}")
                    # print()

                    # Save the best model for each window size and granularity
                    if metrics[0] > best_accuracy:
                        best_accuracy = metrics[0]
                        best_model = model  # the type name of the model: KNN DT LSTM
                        best_metrics = metrics
                        best_m = m
                # Compute and print summary statistics
                average_metrics = np.mean(all_metrics, axis=0)
                print(f"Summary for {model} model on dataset with {gran} granularity and {window} window size:")
                print(
                    f"Average accuracy: {average_metrics[0]} | Average precision: {average_metrics[1]} | Average recall: {average_metrics[2]} | Average f1: {average_metrics[3]}")

            # Save the results of best model for each each window size and granularity
            run_dir = f"best/{gran}_{window}_{model}_repeat{repeat}"
            os.makedirs(run_dir, exist_ok=True)
            save_metrics(best_metrics, os.path.join(run_dir, 'metrics.pkl'))
            save_model(best_m, os.path.join(run_dir, 'model.pkl'))
            # print(f"Results saved in {run_dir}")
            # print()
            print("====================================")
            print(f"The best model for {gran} granularity and {window} window size is {best_model}")
            print(
                f"Accuracy: {best_metrics[0]} | Pecision: {best_metrics[1]} | Recall: {best_metrics[2]} | F1: {best_metrics[3]}")
            print("====================================")


def experiment(model_name, p, data):
    results = None
    if model_name == 'LSTM':
        model, results, _ = trainLSTM(data, hidden_size=p['hidden_size'], dropout=p['drop_out'],
                                      lr=p['lr'], output=False, epochs=10)
    if model_name == 'KNN':
        model, results = trainKNN(data, leaf_size=p['leaf_size'], weights=p['weights'], metric=p['metric'],
                                  output=False, epochs=10)
    if model_name == 'DT':
        model, results = trainDT(data, max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'],
                                 min_samples_split=p['min_samples_split'], output=False, epochs=10)
    return model, results


if __name__ == '__main__':
    _, data_resampled = process_data()
    granularities = ['100ms', '500ms', '1s']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data_level = 'measurement'
    models = ['LSTM', 'DT']
    if os.path.isfile(f'Results/paramsForModels_{"_".join(models)}.pkl'):
        params = pickle.load(open(f'Results/paramsForModels_{"_".join(models)}.pkl', 'rb'))
    else:
        # We only tune on 100ms granularity and 10 window/sequence (see tuneHyperParameters.py)
        # We could tune on different data_level, but i think we should just stick to measurement and use the same params
        # for activity
        params = tuneHyperParameters(models, sensors, 'measurement', data_resampled['100ms'], True)
    windows = [5, 10, 50]
    with open('output_measurement_lstm.txt', 'w') as f:
        sys.stdout = f
        experiments(models, params, data_resampled, granularities, windows, sensors, data_level)

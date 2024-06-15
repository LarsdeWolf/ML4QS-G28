from train_DT import train as trainDT
from train_KNN import train as trainKNN
from train_LSTM import train as trainLSTM
from utils import *
import os
import numpy as np
import pickle
import torch

def save_metrics(metrics, path):
    with open(path, 'wb') as f:
        pickle.dump(metrics, f)

def save_model(model, path):
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f)

def experiments(models, data, granularities, windows, sensors, data_level, repeats=3):
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
                    if model in ['KNN', 'DT']:  # Dataset for KNN & DT are the same and can be re-used
                        if dataset_saved is None:  # If not saved, we get data and save
                            dataset = get_data(data_gran, sensors, data_level, model, window, True, False)
                            dataset_saved = dataset
                        else:  # Using saved dataset
                            dataset = dataset_saved
                    else:  # LSTM get data
                        dataset = get_data(data_gran, sensors, data_level, model, window, True, False)

                    all_metrics = [] # Record each repeat metrics
                    for repeat in range(repeats):
                        print(f"Training {model} model on dataset with {gran} granularity and {window} window size ")
                        print(f'Repeat {repeat}')
                        m, metrics = experiment(model, dataset)
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
                            best_model = model # the type name of the model: KNN DT LSTM
                            best_metrics = metrics
                            best_m = m
                    # Compute and print summary statistics
                    average_metrics = np.mean(all_metrics, axis=0)
                    print(f"Summary for {model} model on dataset with {gran} granularity and {window} window size:")
                    print(f"Average accuracy: {average_metrics[0]} | Average precision: {average_metrics[1]} | Average recall: {average_metrics[2]} | Average f1: {average_metrics[3]}" )

                # Save the results of best model for each each window size and granularity
                run_dir = f"best/{gran}_{window}_{model}_repeat{repeat}"
                os.makedirs(run_dir, exist_ok=True)
                save_metrics(best_metrics, os.path.join(run_dir, 'metrics.pkl'))
                save_model(best_m, os.path.join(run_dir, 'model.pkl'))
                # print(f"Results saved in {run_dir}")
                # print()
                print("====================================")
                print(f"The best model for {gran} granularity and {window} window size is {best_model}")
                print(f"Accuracy: {best_metrics[0]} | Pecision: {best_metrics[1]} | Recall: {best_metrics[2]} | F1: {best_metrics[3]}")
                print("====================================")
def experiment(model_name, data):
    results = None
    if model_name == 'LSTM':
        # model, results, _ = trainLSTM(data, output=False, epochs=10)  # Suppressing train printing output in order to show summarized output (TODO)
        model, results, _ = trainLSTM(data, output=False, epochs=1)
    if model_name == 'KNN':
        model, results = trainKNN(data, output=False, epochs=10)
    if model_name == 'DT':
        model, results = trainDT(data, output=False, epochs=10)
    return model, results

if __name__ == '__main__':
    _, data_resampled = process_data()
    # granularities = ['100ms', '500ms', '1s']
    granularities = ['1s']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data_level = 'measurement'
    models = ['KNN', 'LSTM', 'DT']
    # windows = [5, 10, 50]
    windows = [10, 50]
    experiments(models, data_resampled, granularities, windows, sensors, data_level)

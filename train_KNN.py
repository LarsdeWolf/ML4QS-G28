import load_data
import torch
import numpy as np
import pandas as pd
from features import *
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']


def train(data, epochs=10):
    """
    Trains a KNN model on the data
    Evaluates each epoch on DEV set and saves the best model
    Args:
        data: list of X_train, y_train, X_test, y_test, X_val, y_val
        epochs: Number of epochs to train

    Returns:
        Best model evaluated on the DEV set
    """
    X_train, y_train, X_test, y_test, X_val, y_val = data

    best_model = None
    best_accuracy = 0
    for epoch in range(epochs):
        print("#######################TRAIN#######################")
        # Initialize the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the model
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Epoch {epoch + 1}/{epochs} - Train Accuracy: {train_acc}")

        # Evaluate the model on the DEV set
        print("#######################DEV#######################")
        y_dev_pred = knn.predict(X_val)
        dev_accuracy = accuracy_score(y_val, y_dev_pred)
        print(f"Epoch {epoch + 1}/{epochs} - Dev Accuracy: {dev_accuracy}")

        if dev_accuracy > best_accuracy:  # save best model based on dev accuracy
            best_accuracy = dev_accuracy
            best_model = knn
    # Evaluate best model on the test set
    print("#######################TESTING#######################")
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")

    return best_model


if __name__ == '__main__':
    dataset_level = 'measurement'  # Or activity
    _, data_resampled = load_data.process_data()
    data = data_resampled['100ms']
    if dataset_level == 'measurement':
        X, y = extract_features(data, sensors, multi_processing=True)
        data = train_test_split_measurementlevel(X, y)
    else:
        data_train, data_test, data_dev = train_test_split_activitylevel(data)
        data = [
            *extract_features(data_train, sensors, multi_processing=True, restart=True),  # Train
            *extract_features(data_test, sensors, multi_processing=True, restart=True),  # Test
            *extract_features(data_dev, sensors, multi_processing=True, restart=False)  # Dev
        ]

    model = train(data, epochs=10)
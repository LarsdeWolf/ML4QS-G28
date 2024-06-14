import load_data
import numpy as np
from features import *
from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']


def train(data, epochs=10):
    """
    Trains a Decision Tree model on the data
    Evaluates each epoch on DEV set and saves the best model
    Args:
        data: list of X_train, y_train, X_test, y_test, X_val, y_val
        epochs: Number of epochs to train

    Returns:
        Best model evaluated on the DEV set
    """
    # imputer = SimpleImputer(strategy='mean')    # TODO: need to be replaced, just use a simple one to test
    # X = imputer.fit_transform(X)

    X_train, y_train, X_test, y_test, X_val, y_val = data

    best_model = None
    best_accuracy = 0

    for epoch in range(epochs):
        print("#######################TRAIN#######################")
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=4)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc}")

        print("#######################DEV#######################")
        y_dev_pred = model.predict(X_val)
        dev_accuracy = accuracy_score(y_val, y_dev_pred)
        print(f"Epoch {epoch + 1}/{epochs} - Dev Accuracy: {dev_accuracy}")

        if dev_accuracy > best_accuracy:  # save best model based on dev accuracy
            best_accuracy = dev_accuracy
            best_model = model

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
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = get_data(data, sensors, dataset_level, 'DT', 10)
    model = train(data, epochs=10)

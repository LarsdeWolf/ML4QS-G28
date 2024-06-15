from __future__ import print_function
import builtins
from load_data import process_data
from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

debug = True
def print(*args, **kwargs):
     if(debug):
             return builtins.print(*args, **kwargs)


sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']


def train(data, output=True, epochs=10):
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

    global debug
    debug = output

    for epoch in range(epochs):
        print("#######################TRAIN#######################")
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=4)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1, train_precision, train_recall = (f1_score(y_train, y_train_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
                                                   precision_score(y_train, y_train_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
                                                   recall_score(y_train, y_train_pred, list(label_to_id.values()), average='micro', zero_division=0.0))
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc} | Train Precision: {train_precision} | Train Recall: {train_recall} | Train F1: {train_f1}")

        print("#######################DEV#######################")
        y_dev_pred = model.predict(X_val)
        dev_accuracy = accuracy_score(y_val, y_dev_pred)
        dev_f1, dev_precision, dev_recall = (
            f1_score(y_val, y_dev_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
            precision_score(y_val, y_dev_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
            recall_score(y_val, y_dev_pred, list(label_to_id.values()), average='micro', zero_division=0.0))
        print(f"Epoch {epoch + 1}/{epochs} - Dev Accuracy: {dev_accuracy} | Dev Precision: {dev_precision} | Dev Recall: {dev_recall} | Dev F1: {dev_f1}")

        if dev_accuracy > best_accuracy:  # save best model based on dev accuracy
            best_accuracy = dev_accuracy
            best_model = model

    # Evaluate best model on the test set
    print("#######################TESTING#######################")
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1, test_precision, test_recall = (
        f1_score(y_test, y_test_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
        precision_score(y_test, y_test_pred, list(label_to_id.values()), average='micro', zero_division=0.0),
        recall_score(y_test, y_test_pred, list(label_to_id.values()), average='micro', zero_division=0.0))
    print(f"Test Accuracy: {test_accuracy} | Test Precision: {test_precision} | Test Recall: {test_recall} | Test F1: {test_f1}")

    return best_model


if __name__ == '__main__':
    dataset_level = 'measurement'  # Or activity
    _, data_resampled = process_data()
    data = data_resampled['100ms']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = get_data(data, sensors, dataset_level, 'DT', 10, True, True)
    model = train(data, output=True, epochs=1)

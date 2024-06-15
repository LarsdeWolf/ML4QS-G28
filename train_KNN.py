from __future__ import print_function
import builtins
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from load_data import process_data
from utils import *
from cleaning import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

debug = True
def print(*args, **kwargs):
     if(debug):
             return builtins.print(*args, **kwargs)


def train(data, output=True, epochs=10):
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

    global debug
    debug = output

    for epoch in range(epochs):
        print("#######################TRAIN#######################")
        # Initialize the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the model
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0.0)
        train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0.0)
        train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0.0)
        print(f"Epoch {epoch + 1}/{epochs} - Train Accuracy: {train_acc} | Train Precision: {train_precision} | Train Recall: {train_recall} | Train F1: {train_f1}")

        # Evaluate the model on the DEV set
        print("#######################DEV#######################")
        y_dev_pred = knn.predict(X_val)
        dev_accuracy = accuracy_score(y_val, y_dev_pred)
        dev_f1 = f1_score(y_val, y_dev_pred, average='macro', zero_division=0.0)
        dev_precision = precision_score(y_val, y_dev_pred, average='macro', zero_division=0.0)
        dev_recall = recall_score(y_val, y_dev_pred, average='macro', zero_division=0.0)
        print(f"Epoch {epoch + 1}/{epochs} - Dev Accuracy: {dev_accuracy} | Dev Precision: {dev_precision} | Dev Recall: {dev_recall} | Dev F1: {dev_f1}")

    if dev_accuracy > best_accuracy:  # save best model based on dev accuracy
            best_accuracy = dev_accuracy
            best_model = knn
    # Evaluate best model on the test set
    print("#######################TESTING#######################")
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0.0)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0.0)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0.0)
    print(f"Test Accuracy: {test_accuracy} | Test Precision: {test_precision} | Test Recall: {test_recall} | Test F1: {test_f1}")

    # plot_confusion_matrix(y_test, y_test_pred, title='Test Set Confusion Matrix')

    return best_model


if __name__ == '__main__':
    dataset_level = 'measurement'  # Or activity
    # dataset_level = 'activity'
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    _, data_resampled = process_data()
    data = data_resampled['100ms']
    data = clean_data(data)
    data = get_data(data, sensors, dataset_level, 'KNN', 10, True, True)
    model = train(data, epochs=10)

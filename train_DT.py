import load_data
import numpy as np
from features import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


label_to_id = {'walk': 0, 'run': 1, 'bike': 2, 'car': 3, 'train': 4}
sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']

def train_test_split_custom(X, y, train_size=0.75, test_size=0.15, dev_size=0.10):
    """
    Shuffles and splits the datapoints into training, testing and validation sets
    """
    perm = np.random.permutation(len(X))
    train_split, test_split, dev_split = int(len(X) * train_size), int(len(X) * test_size), int(len(X) * dev_size)

    X_train, y_train = X[perm[:train_split]], y[perm[:train_split]]
    X_test, y_test = X[perm[train_split:train_split + test_split]], y[perm[train_split:train_split + test_split]]
    X_dev, y_dev = X[perm[train_split + test_split:len(X)]], y[perm[train_split + test_split:len(X)]]

    return X_train, y_train, X_test, y_test, X_dev, y_dev


def train(data, epochs=10):
    """
    Trains a Decision Tree model on the data
    Evaluates each epoch on DEV set and saves the best model
    Args:
        data: Output of load_data.process_data() (normal data or resampled data)
        epochs: Number of epochs to train

    Returns:
        Best model evaluated on the DEV set
    """
    X, y = extract_features(data, sensors, multi_processing=True)
    imputer = SimpleImputer(strategy='mean')    # TODO: need to be replaced, just use a simple one to test
    X = imputer.fit_transform(X)

    X_train, y_train, X_test, y_test, X_val, y_val = train_test_split_custom(X, y)

    best_model = None
    best_accuracy = 0

    for epoch in range(epochs):
        print("#######################TRAIN#######################")
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=4)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {val_accuracy}")

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
    _, data_resampled = load_data.process_data()
    data = data_resampled['100ms']
    model = train(data, epochs=10)
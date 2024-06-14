from load_data import *
from features import *
from train_DT import *
from train_KNN import *
from train_LSTM import *
from utils import *



def experiments(models, data, sensors, data_level):
    if data_level == 'measurement':
        X_train, y_train, X_test, y_test, X_val, y_val = train_test_split_measurementlevel(
            extract_features(data, sensors, multi_processing=True))
    else:
        data_train, data_test, data_dev = train_test_split_activitylevel(data)
        X_train, y_train = extract_features(data_train, sensors, multi_processing=True, restart=True)
        X_test, y_test = extract_features(data_test, sensors, multi_processing=True, restart=True)
        X_dev, y_dev = extract_features(data_dev, sensors, multi_processing=True, restart=False)



_, data_resampled = process_data()
granularities = ['100ms', '500ms', '1s']
sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
data_level = 'measurement'
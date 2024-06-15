import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to clean data using LOF outlier removal and interpolation
def clean_data(df, contamination=0.1, n_neighbors=20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    # Initial imputation of missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Dynamically set n_neighbors based on the size of the dataset
    for col in numeric_cols:
        data_col = df[col].values.reshape(-1, 1)
        actual_n_neighbors = min(n_neighbors, len(data_col) - 1)

        # Apply LOF for outlier detection
        lof = LocalOutlierFactor(n_neighbors=actual_n_neighbors, contamination=contamination)
        is_outlier = lof.fit_predict(data_col)
        df.loc[is_outlier == -1, col] = np.nan  # Mark outliers as NaN

    # Final interpolation of missing values
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    return df

# Plotting function
def plot_data(original, cleaned, title):
    plt.figure(figsize=(20, 5))
    time_col = original['timestamp'] if 'timestamp' in original.columns else range(len(original))
    for i, col in enumerate(cleaned.columns):
        plt.subplot(1, len(cleaned.columns), i + 1)
        plt.plot(time_col, original[col], label='Original', alpha=0.5)
        plt.plot(time_col, cleaned[col], label='Cleaned', alpha=0.5)
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.show()

# Directory path
base_path = '/Users/tangzj/Desktop/ML4QS/ML4QS-G28/Data/bike Pocket(Vaji)2'

# Filenames
file_names = [
    'Accelerometer.csv',
    'Gyroscope.csv',
    'Linear Accelerometer.csv',
    'Location.csv',
    'Proximity.csv'
]

# Process each file
for file_name in file_names:
    file_path = base_path + file_name
    data = load_data(file_path)
    original_data = data.copy()
    cleaned_data = clean_data(data)
    plot_data(original_data, cleaned_data, file_name)
    # cleaned_data.to_csv(base_path + 'cleaned_' + file_name, index=False)
    # print(f'Cleaned data for {file_name} saved.')

print('All files have been cleaned and saved.')

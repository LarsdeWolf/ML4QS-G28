import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load CSV data from a file path."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clean_data(df, contamination=0.1, n_neighbors=20):
    """Clean data by removing outliers using LOF and interpolating missing values."""
    if df is None:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    # Initial imputation of missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Apply LOF for outlier detection
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

def plot_data(original, cleaned, title):
    """Plot original and cleaned data."""
    if original is None or cleaned is None:
        return

    plt.figure(figsize=(12, 5))
    if 'timestamp' in original.columns:
        time_col = original['timestamp']
    else:
        time_col = range(len(original))

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols):
        plt.subplot(1, len(numeric_cols), i + 1)
        plt.plot(time_col, original[col], label='Original', alpha=0.5)
        plt.plot(time_col, cleaned[col], label='Cleaned', alpha=0.75)
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Directory path for the dataset
base_path = '/Users/vaji/Documents/ProjectMealDeliveryJ1/ML4QS-G28/Data/train pocket (Lars)/'

# Names of the files
file_names = ['Accelerometer.csv', 'Gyroscope.csv', 'Linear Accelerometer.csv']

# Process each file
for file_name in file_names:
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        data = load_data(file_path)
        if data is not None:
            original_data = data.copy()
            cleaned_data = clean_data(data)
            plot_data(original_data, cleaned_data, file_name)
            cleaned_data.to_csv(os.path.join(base_path, f'cleaned_{file_name}'), index=False)
            print(f'Cleaned data for {file_name} saved.')
    else:
        print(f"File not found: {file_name}")

print('Processing complete.')

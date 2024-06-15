import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer, KNNImputer

def load_data(file_path):
    """Load CSV data from a file path."""
    return pd.read_csv(file_path)

def initial_impute(df, numeric_cols):
    """Initial imputation to handle NaN values before LOF."""
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def clean_data(df):
    """Clean data by removing outliers using Local Outlier Factor and interpolating missing values using KNN imputation."""
    # Identify numeric columns, avoid 'timestamp' if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    # Initial imputation to handle NaN values before applying LOF
    df = initial_impute(df, numeric_cols)

    # Outlier detection using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.03)  # Adjusted parameters
    outliers = lof.fit_predict(df[numeric_cols])
    df.loc[outliers == -1, numeric_cols] = np.nan  # Mark outliers as NaN

    # Imputation for missing values using KNN
    imputer = KNNImputer(n_neighbors=10)  # Adjusted number of neighbors
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def plot_data(original, cleaned, title):
    """Plot original and cleaned data."""
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
        original_data = data.copy()
        cleaned_data = clean_data(data)
        plot_data(original_data, cleaned_data, file_name)
        cleaned_data.to_csv(os.path.join(base_path, f'cleaned_{file_name}'), index=False)
        print(f'Cleaned data for {file_name} saved.')
    else:
        print(f"File not found: {file_name}")

print('Processing complete.')

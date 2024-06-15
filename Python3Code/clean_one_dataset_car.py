import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to clean data using distance-based outlier removal and interpolation
def clean_data(df):
    # Identify numeric columns, avoiding the 'timestamp' if it exists
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    # Apply distance-based outlier detection
    for col in numeric_cols:
        distance = np.abs(df[col] - df[col].mean())
        threshold = distance.std() * 3
        outliers = distance > threshold
        df.loc[outliers, col] = np.nan  # Mark outliers as NaN

    # Interpolate missing values
    df[numeric_cols] = df[numeric_cols].interpolate()

    return df

# Function to plot original and cleaned data
def plot_data(original, cleaned, title):
    plt.figure(figsize=(12, 5))
    if 'timestamp' in original.columns:
        time_col = original['timestamp']
    else:
        time_col = range(len(original))  # Use index if no timestamp available

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cols_to_plot = [col for col in numeric_cols if col != 'timestamp']
    num_plots = len(cols_to_plot)  # Adjust the number of plots dynamically

    for i, col in enumerate(cols_to_plot):
        plt.subplot(1, num_plots, i + 1)  # Ensure the subplot index is valid
        plt.plot(time_col, original[col], label='Original', alpha=0.5)
        plt.plot(time_col, cleaned[col], label='Cleaned', alpha=0.5)
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Base directory path
base_path = '/Users/vaji/Documents/ProjectMealDeliveryJ1/ML4QS-G28/Data/car pocket (Lars)/'

# File names
file_names = [
    'Accelerometer.csv',
    'Gyroscope.csv',
    'Linear Accelerometer.csv',
    'Location.csv',
    'Proximity.csv'
]

# Check and process each file if it exists
for file_name in file_names:
    file_path = base_path + file_name
    if os.path.exists(file_path):
        print("Attempting to load:", file_path)
        data = load_data(file_path)
        original_data = data.copy()
        cleaned_data = clean_data(data)
        plot_data(original_data, cleaned_data, file_name)
        # Save cleaned data back to CSV
        cleaned_data.to_csv(base_path + 'cleaned_' + file_name, index=False)
        print(f'Cleaned data for {file_name} saved.')
    else:
        print(f"File not found: {file_name}")

print('Processed all available files.')

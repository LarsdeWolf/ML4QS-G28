import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to clean data using distance-based outlier removal and interpolation
def clean_data(df):
    # Assuming the data is numerical and might or might not have a 'timestamp' column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')  # Remove timestamp from numeric columns if present

    # Apply a simple distance-based outlier detection
    for col in numeric_cols:
        distance = np.abs(df[col] - df[col].mean())
        threshold = distance.std() * 3
        outliers = distance > threshold
        df.loc[outliers, col] = np.nan  # Mark outliers as NaN

    # Interpolate missing values
    df[numeric_cols] = df[numeric_cols].interpolate()

    return df

# Plotting function
def plot_data(original, cleaned, title):
    plt.figure(figsize=(15, 5))
    if 'timestamp' in original.columns:
        time_col = original['timestamp']
    else:
        time_col = range(len(original))  # Use index if no timestamp available

    for i, col in enumerate(cleaned.columns):
        plt.subplot(1, len(cleaned.columns), i + 1)
        plt.plot(time_col, original[col], label='Original', alpha=0.5)
        plt.plot(time_col, cleaned[col], label='Cleaned', alpha=0.5)
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.show()

# Directory path
base_path = '/Users/vaji/Documents/ProjectMealDeliveryJ1/ML4QS-G28/Data/bike Pocket(Vaji)1/'

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
    # Save cleaned data back to CSV
    cleaned_data.to_csv(base_path + 'cleaned_' + file_name, index=False)
    print(f'Cleaned data for {file_name} saved.')

print('All files have been cleaned and saved.')

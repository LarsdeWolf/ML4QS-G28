import os
import pandas as pd
from glob import glob


# Load From .Data
def process_data(path: str = './Data',          # Path to data
                 granularities: list = None):   # Granularity in seconds
    """
    Process data by converting each measurement in ./Data to a DataFrame and resampling it to the specified granularity
    Assumes that each measurement is in a separate directory and that the directory name is the activity label

    Args:
        path: path to measurement dir
        granularities: list of granularities to resample to  (See Book Chapter 2.2)

    Returns:
        data: list of DataFrames for each measurement (unresampled)
        data_resampled: dict with granularity as key and list of resampled DataFrames as values

    """
    if granularities is None:
        granularities = ['100ms', '500ms', '1s', '10s', '30s']
    raw_data = [measurement_to_df(path + '/' + record, record.split()[0], count) for count, record in enumerate(os.listdir(path))]
    data = [df for df in raw_data if df is not None]  # Filter out None values
    
    data_resampled = {}
    for granularity in granularities:
        for measurement in data:
            if measurement is not None:
                if granularity not in data_resampled:
                    data_resampled[granularity] = [measurement.resample(granularity).mean()]
                    continue
                data_resampled[granularity].append(measurement.resample(granularity).mean())

    return data, data_resampled


def measurement_to_df(measurement: str, activity: str, count: int):
    """
    Converts a measurement to a DataFrame by merging all sensor dataframes on the 'Time (s)' column
    Adds columns Time (ns), and one-hot label columns 'walk', 'run', 'car', 'train'
    Args:
        measurement: path to measurement dir
        activity: type of activity during measurement
        count: integer to seperate measurements (e.g. sepearting 2 car experiments)

    Returns:
        df: DataFrame of measurement data with added columns

    """
    df = None
    for sensor in glob(f'{measurement}/*.csv'):
        s_name = os.path.basename(sensor).split('.')[-2] # Get sensor name
        if s_name == 'Proximity':
            continue
        df_ = pd.read_csv(sensor)
        df_['Time (s)'] = pd.to_numeric(df_['Time (s)'], errors='coerce')  # Ensure 'Time (s)' is numeric
        df_ = df_.dropna(subset=['Time (s)'])  # Drop rows where 'Time (s)' is NaN
        df_['Time (s)'] = df_['Time (s)'].astype(float)  # Ensure 'Time (s)' is float64
        df_.rename(columns=lambda x: f"{s_name}_{x}" if x != 'Time (s)' else x, inplace=True)   # Rename columns
        df_.rename(columns=lambda x: x.replace('Linear Accelerometer', 'Lin-Acc') if 'Linear Accelerometer' in x
            else x, inplace=True)
        
        if df is None:
            df = df_
            continue
        
        df = pd.merge_asof(df, df_, on='Time (s)', direction='nearest')
        
    if df is None:
        print(f"Warning: No sensor data found in {measurement}")
        return None
    
    for coll in ['walk', 'run', 'bike', 'car', 'train']:
        df[coll] = 1 if coll == activity else 0                                                 # One-hot encode activity
    df['id'] = count
    df['Time (ns)'] = (df['Time (s)'] * 1e9).astype('int64')                                    # Convert to ns
    df['Time (s)'] = pd.to_datetime(df['Time (s)'], unit='s')
    df.set_index('Time (s)', inplace=True)
    return df


if __name__ == '__main__':
    data, resampled = process_data()
    print(f"Processed {len(data)} measurements.")
    for granularity, resampled_data in resampled.items():
        print(f"Granularity: {granularity}, Number of resampled DataFrames: {len(resampled_data)}")


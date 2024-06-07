from load_data import *
import matplotlib.pyplot as plt
import pandas as pd


def plot_sensor(df: pd.DataFrame, sensor: str):
    """
    Plots the sensor data for a given DataFrame
    Args:
        df: DataFrame with sensor data
        sensor: sensor to plot
    """
    plt.figure(figsize=(20, 5))
    columns = [col for col in df.columns if sensor in col]
    plt.plot(df.index, df[columns])
    plt.xlabel('Time (s)')
    plt.ylabel(sensor)
    plt.title(f'{sensor} over time')
    plt.legend(columns)
    plt.show()

def statistics_over_set(data: list, sensor: str, activity: str):
    """
    Prints statistics over a set of data for a given sensor and activity
    Returns dataframe, where index (e.g. result['mean']) prints the statistic for each dataset,
    only showing the columns related to the sensor. Only shows the data for the given activity
    Args:
        data: list of DataFrames
        sensor: sensor to describe
        activity: type of activity (e.g. walk, run, etc.)
    """
    columns = [col for col in data[0].columns if sensor in col]
    data_activity = [df for df in data if df[activity].all() == 1]  # Only show data for selected activity
    desc_dfs = []
    for i, df in enumerate(data_activity):
        desc_df = df[columns].describe().T
        desc_df.columns = pd.MultiIndex.from_product([[f'd{i + 1}'], desc_df.columns])
        desc_dfs.append(desc_df)

    result = pd.concat(desc_dfs, axis=1)
    result.columns = result.columns.swaplevel(0, 1)
    result.sort_index(axis=1, level=[0, 1], inplace=True)
    return result




if __name__ == '__main__':
    data, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin_Acc', 'Gyroscope', 'Location', 'Proximity']
    activities = ['walk', 'run', 'bike', 'car', 'train']
    for sensor in sensors:
        plot_sensor(data[0], sensor)  # Only plotting for the first dataset
        result = statistics_over_set(data, sensor, activities[0])  # Only printing statistics for 'walk' datasets
        print(result['mean'])


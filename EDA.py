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


if __name__ == '__main__':
    data, data_resampled = process_data()
    sensors = ['Accelerometer', 'Lin_Acc', 'Gyroscope', 'Location', 'Proximity']
    for sensor in sensors:  # Testing with fist data point
        plot_sensor(data[0], sensor)

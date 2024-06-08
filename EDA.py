from load_data import *
import matplotlib.pyplot as plt
import pandas as pd

sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
activities = ['walk', 'run', 'bike', 'car', 'train']

def plot_set(data: list, sensor: str, activity: str, plot: str = 'lineplot'):
    """
    Plots the sensor data for a given set of data
    Plots the data for each dataset that is labeled with the activity
    Args:
        data:
        sensor:
        activity:
        plot:
    Returns:

    """
    # Creating a figure with subplots
    columns = [col for col in data[0].columns if sensor in col]
    data_activity = [df for df in data if df[activity].all() == 1]
    if len(data_activity) < 2:  # If not all the data is available (yet), use single plots
        plot_single(data_activity[0], sensor);  return
    fig, axs = plt.subplots(len(data_activity), 1, figsize=(20, 5 * len(data_activity)))
    for i, df in enumerate(data_activity):
        if plot == 'lineplot':
            axs[i].plot(df.index, df[columns])
            axs[i].set_xlabel('Time (s)')
            fig.legend(columns, loc='upper right')
        else:
            axs[i].boxplot(df[columns].values, labels=columns)
        axs[i].set_ylabel(sensor)
        axs[i].set_title(f'dataset {df["id"][0]}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f'{plot} of {sensor} over time for {activity} datasets', fontsize=16)
    plt.show()


def plot_single(df: pd.DataFrame, sensor: str, plot: str = 'lineplot'):
    """
    Plots the sensor data for single pandas dataframe (measurement)
    Args:
        df: DataFrame with sensor data
        sensor: sensor to visualize
        plot: plot type (lineplot/boxplot)
    """
    act_name = [act for act in activities if df[act].all() == 1][0]
    plt.figure(figsize=(25, 5))
    columns = [col for col in df.columns if sensor in col]
    if plot == 'lineplot':
        plt.plot(df.index, df[columns])
        plt.xlabel('Time (s)')
        plt.legend(columns, loc='upper right')
    else:  # Plot Boxplot
        plt.boxplot(df[columns].values, labels=columns)
    plt.ylabel(sensor)
    plt.title(f'{plot} of {sensor} over time for dataset {df["id"][0]} ({act_name})')
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
    files = str(os.listdir('./data'))
    data, data_resampled = process_data()

    # # Plotting each sensor for specified dataset (1 sensor per plot)
    # for sensor in sensors:  # Plotting each sensor
    #     plot_single(data[0], sensor, 'lineplot')  # Only plotting for the first dataset (change plot to get boxplots)

    # location is way too different for each record, hard to plot
    for sensor in ['Accelerometer', 'Lin-Acc', 'Gyroscope']:  # Plotting each sensor
        plot_single(data[0], sensor, 'lineplot')  # Only plotting for the first dataset (change plot to get boxplots)

    # Plotting each dataset that belong to the specified activity for the specified sensor
    for activity in activities:
        if activity not in files:
            continue
        plot_set(data, sensors[0], activity, 'lineplot')  # Only using the first sensor (change plot to get boxplots)
        # Printing (mean) statistic(s) for each dataset for the activity
        result = statistics_over_set(data, sensors[0], activity)
        print(activity)
        print(result['mean'])

    # Visualizing the difference between different granularities, for fixed sensor and activity
    plot_set(data, sensors[0], 'car')
    plot_set(data_resampled['10s'], sensors[0], 'car')
    plot_set(data_resampled['500ms'], sensors[0], 'car')






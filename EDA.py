import os
import pandas as pd
import numpy as np
from glob import glob
from typing import Dict, List, Any
from Python3Code.Chapter2.CreateDataset import CreateDataset


# Load From .Data
def load_data(path: str = './Data',
              rename: bool = False,
              merge: bool = False) -> Dict[str, List[Any]]:
    """
    Loads the PhyPhox data by creating a pd.DataFrame for each sensor for each measurement
    Returns a dict with activity type as key and lists of dicts containing the sensory dataframes as values

    path: Path to Data, assumes directories that start with the activity type as name

    """
    def load_measurement(measurement):
        dfs = {}
        for sensor in glob(f'{path}/{measurement}/*.csv'):
            name = os.path.basename(sensor).split('.')[-2]
            df = pd.read_csv(sensor)
            df.rename(columns=lambda x: f"{name}_{x}" if x != 'Time (s)' else x, inplace=True)
            dfs[name] = df
        return dfs

    measurements = {
        'walk': [],
        'run': [],
        'car': [],
        'train': [],
    }

    for run in os.listdir(path):
        data = load_measurement(run)
        name = run.split()[0]
        measurements[name].append(data)

    return measurements

x = load_data()





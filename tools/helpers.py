import yaml
import pickle
import pandas as pd
import os
from tqdm import tqdm
import sys
from pathlib import Path
from fastparquet import write
import numpy as np


def read_yaml(file_path):
    """
    Read a YAML file and return the contents as a dictionary.
    Parameters
    ----------
    file_path : str
        The path of the YAML file to read.
    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        return data


def write_yaml(data, file_path):
    """
    Write a dictionary to a YAML file.
    Ensures all values are in standard Python types before writing.
    
    Parameters
    ----------
    data : dict
        The dictionary to write to a YAML file.
    file_path : str
        The path of the file where the YAML data shall be stored.
    
    Returns
    -------
    None
    """

    # Convert non-standard types (like numpy.float64) to standard Python types
    for key in data:
        value = data[key]
        if isinstance(value, np.float64):
            data[key] = float(value)  # Convert to native Python float
        elif isinstance(value, np.int64):
            data[key] = int(value)  # Convert to native Python int

    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"Data successfully written to YAML at {file_path}")


def update_yaml(file_path, new_items):
    """
    Update a YAML file with the contents of a dictionary.
    Parameters
    ----------
    file_path : str
        The path of the YAML file to update.
    new_items : dict
        The dictionary of new key-value pairs to add to the existing YAML file.
    Returns
    -------
    None
    """
    data = read_yaml(file_path)
    data.update(new_items)
    write_yaml(data, file_path)


def pickle_to_dict(file_path):
    """
    Loads a dictionary from a pickle file at a specified file path.
    Parameters
    ----------
    file_path : str
        The path of the pickle file to load.
    Returns
    -------
    dict
        The dictionary loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def dict_to_pickle(dic, target_path):
    """
    Saves a dictionary to a pickle file at the specified target path.
    Creates target directory if not existing.
    Parameters
    ----------
    dic : dict
        The dictionary to save to a pickle file.
    target_path : str
        The path of the file where the dictionary shall be stored.
    Returns
    -------
    None
    """
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(target_path, 'wb') as f:
        pickle.dump(dic, f)


def drop_keys(dic, keys_to_drop):
    """Removes specified keys from a dictionary.
    Parameters
    ----------
    dict : dict
        The dictionary to remove keys from.
    keys_to_drop : list
        A list of keys to remove from the dictionary.
    Returns
    -------
    dict
        A modified dictionary with the specified keys removed.
    """
    # Create a set of keys to be dropped
    keys_to_drop_set = set(keys_to_drop)
    # Create a new dictionary with all elements from dict except for the ones in keys_to_drop
    new_dict = {key: dic[key] for key in dic.keys() if key not in keys_to_drop_set}
    return new_dict


def parquet_to_dict(directory_path: str, pbar: bool = True) -> dict:
    """
    Recursively loads dataframes from the parquet files in the specified directory and returns a dictionary.
    Nested directories are supported.
    Parameters
    ----------
    directory_path : str
        The directory path containing the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    Returns
    -------
    dict
        A dictionary containing the loaded pandas dataframes.
    """
    dictionary = {}
    if pbar:
        bar_iter = tqdm(sorted(os.listdir(directory_path)), desc='Reading parquet files: ')
    else:
        bar_iter = sorted(os.listdir(directory_path))
    for file_name in bar_iter:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            dictionary[file_name] = parquet_to_dict(file_path, pbar=False)
        elif file_name.endswith(".parquet"):
            k = file_name[:-len(".parquet")]
            dictionary[k] = pd.read_parquet(file_path)
    return dictionary


def dict_to_parquet(dictionary: dict, directory_path: str, pbar: bool = True) -> None:
    """
    Recursively stores the dataframes in the input dictionary as parquet files in the specified directory.
    Nested dictionaries are supported. If the specified directory does not exist, it will be created.
    Parameters
    ----------
    dictionary : dict
        A nested dictionary containing pandas dataframes.
    directory_path : str
        The directory path to store the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if pbar:
        bar_iter = tqdm(dictionary.items(), desc='Writing parquet files: ')
    else:
        bar_iter = dictionary.items()
    for k, v in bar_iter:
        if isinstance(v, dict):
            dict_to_parquet(v, os.path.join(directory_path, k), pbar=False)
        else:
            file_path = os.path.join(directory_path, k + ".parquet")
            write(file_path, v, compression='GZIP')


matilda_vars = {
    'avg_temp_catchment': ('Mean Catchment Temperature', '°C'),
    'avg_temp_glaciers': ('Mean Temperature of Glacierized Area', '°C'),
    'evap_off_glaciers': ('Off-glacier Evaporation', 'mm w.e.'),
    'prec_off_glaciers': ('Off-glacier Precipitation', 'mm w.e.'),
    'prec_on_glaciers': ('On-glacier Precipitation', 'mm w.e.'),
    'rain_off_glaciers': ('Off-glacier Rain', 'mm w.e.'),
    'snow_off_glaciers': ('Off-glacier Snow', 'mm w.e.'),
    'rain_on_glaciers': ('On-glacier Rain', 'mm w.e.'),
    'snow_on_glaciers': ('On-glacier Snow', 'mm w.e.'),
    'snowpack_off_glaciers': ('Off-glacier Snowpack', 'mm w.e.'),
    'soil_moisture': ('Soil Moisture', 'mm w.e.'),
    'upper_groundwater': ('Upper Groundwater', 'mm w.e.'),
    'lower_groundwater': ('Lower Groundwater', 'mm w.e.'),
    'melt_off_glaciers': ('Off-glacier Melt', 'mm w.e.'),
    'melt_on_glaciers': ('On-glacier Melt', 'mm w.e.'),
    'ice_melt_on_glaciers': ('On-glacier Ice Melt', 'mm w.e.'),
    'snow_melt_on_glaciers': ('On-glacier Snow Melt', 'mm w.e.'),
    'refreezing_ice': ('Refreezing Ice', 'mm w.e.'),
    'refreezing_snow': ('Refreezing Snow', 'mm w.e.'),
    'total_refreezing': ('Total Refreezing', 'mm w.e.'),
    'SMB': ('Glacier Surface Mass Balance', 'mm w.e.'),
    'actual_evaporation': ('Mean Actual Evaporation', 'mm w.e.'),
    'total_precipitation': ('Mean Total Precipitation', 'mm w.e.'),
    'total_melt': ('Total Melt', 'mm w.e.'),
    'runoff_without_glaciers': ('Runoff without Glaciers', 'mm w.e.'),
    'runoff_from_glaciers': ('Runoff from Glaciers', 'mm w.e.'),
    'total_runoff': ('Total Runoff', 'mm w.e.'),
    'glacier_area': ('Glacier Area', 'km²'),
    'glacier_elev': ('Mean Glacier Elevation', 'm.a.s.l.'),
    'smb_water_year': ('Surface Mass Balance of the Hydrological Year', 'mm w.e.'),
    'smb_scaled': ('Area-scaled Surface Mass Balance', 'mm w.e.'),
    'smb_scaled_capped': ('Surface Mass Balance Capped at 0', 'mm w.e.'),
    'smb_scaled_capped_cum': ('Cumulative Surface Mass Balance Capped at 0', 'mm w.e.'),
    'glacier_melt_perc': ('Melted Glacier Fraction', '%'),
    'glacier_mass_mmwe': ('Glacier Mass', 'mm w.e.'),
    'glacier_vol_m3': ('Glacier Volume', 'm³'),
    'glacier_vol_perc': ('Fraction of Initial Glacier Volume (2000)', '-')
}


def water_year(df, begin=10):
    """
    Calculates the water year for each date in the index of the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    begin : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    numpy.ndarray
        An array of integers representing the water year for each date in the input DataFrame index.
    """
    return np.where(df.index.month < begin, df.index.year, df.index.year + 1)


def crop2wy(df, begin=10):
    """
    Crops a DataFrame to include only the rows that fall within a complete water year.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex and a 'water_year' column.
    begin : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    pandas.DataFrame or None
        A new DataFrame containing only the rows that fall within a complete water year.
    """
    cut_begin = pd.to_datetime(f'{begin}-{df.water_year.iloc[0]}', format='%m-%Y')
    cut_end = pd.to_datetime(f'{begin}-{df.water_year.iloc[-1] - 1}', format='%m-%Y') - pd.DateOffset(days=1)
    return df[cut_begin:cut_end].copy()


def hydrologicalize(df, begin_of_water_year=10):
    """
    Adds a 'water_year' column to a DataFrame and crops it to include only complete water years.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    begin_of_water_year : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    pandas.DataFrame or None
        A new DataFrame with a 'water_year' column and only rows that fall within complete water years.
    """
    df_new = df.copy()
    df_new['water_year'] = water_year(df_new, begin_of_water_year)
    return crop2wy(df_new, begin_of_water_year)

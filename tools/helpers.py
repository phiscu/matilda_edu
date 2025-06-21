import yaml
import pickle
import pandas as pd
import os
import sys
import numpy as np
import spotpy
import contextlib
from pathlib import Path
from fastparquet import write
from bias_correction import BiasCorrection
from tqdm import tqdm
from matilda.core import matilda_simulation
from multiprocessing import Pool
from functools import partial


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


def adjust_jupyter_config():
    from jupyter_server import serverapp
    from dash._jupyter import _jupyter_config
    import os

    js = list(serverapp.list_running_servers())[0]

    if js['hostname'] == 'localhost':
        print('JupyterLab seems to run on local machine.')
    else:
        base = js['base_url']
        if base.split('/')[1] == 'binder':
            print('JupyterLab seems to run on binder server.')

            # start updating jupyter server config
            # official docu: https://dash.plotly.com/dash-in-jupyter
            # however, due to problems of jupyterlab v4 a work-around must be implemented
            # see: https://github.com/plotly/dash/issues/2804
            # and: https://github.com/plotly/dash/issues/2998
            # solution inspired by: https://github.com/mthiboust/jupyterlab-retrieve-base-url/tree/main
            conf = {'type': 'base_url_response',
                    'server_url': 'https://notebooks.gesis.org',
                    'base_subpath': os.getenv('JUPYTERHUB_SERVICE_PREFIX'),
                    'frontend': 'jupyterlab'}

            _jupyter_config.update(conf)
            print('Jupyter config has been updated to run Dash!')
        else:
            print('JupyterLab seems to run on unsupported environment.')


class DataFilter:
    def __init__(self, df, zscore_threshold=3, resampling_rate=None, prec=False, jump_threshold=5):
        self.df = df
        self.zscore_threshold = zscore_threshold
        self.resampling_rate = resampling_rate
        self.prec = prec
        self.jump_threshold = jump_threshold
        self.filter_all()


    def check_outliers(self):
        """
        A function for filtering a pandas dataframe for columns with obvious outliers
        and dropping them based on a z-score threshold.

        Returns
        -------
        models : list
            A list of columns identified as having outliers.
        """
        # Resample if rate specified
        if self.resampling_rate is not None:
            if self.prec:
                self.df = self.df.resample(self.resampling_rate).sum()
            else:
                self.df = self.df.resample(self.resampling_rate).mean()

        # Calculate z-scores for each column
        z_scores = pd.DataFrame((self.df - self.df.mean()) / self.df.std())

        # Identify columns with at least one outlier (|z-score| > threshold)
        cols_with_outliers = z_scores.abs().apply(lambda x: any(x > self.zscore_threshold))
        self.outliers = list(self.df.columns[cols_with_outliers])

        # Return the list of columns with outliers
        return self.outliers

    def check_jumps(self):
        """
        A function for checking a pandas dataframe for columns with sudden jumps or drops
        and returning a list of the columns that have them.

        Returns
        -------
        jumps : list
            A list of columns identified as having sudden jumps or drops.
        """
        cols = self.df.columns
        jumps = []

        for col in cols:
            diff = self.df[col].diff()
            if (abs(diff) > self.jump_threshold).any():
                jumps.append(col)

        self.jumps = jumps
        return self.jumps

    def filter_all(self):
        """
        A function for filtering a dataframe for columns with obvious outliers
        or sudden jumps or drops in temperature, and returning a list of the
        columns that have been filtered using either or both methods.

        Returns
        -------
        filtered_models : list
            A list of columns identified as having outliers or sudden jumps/drops in temperature.
        """
        self.check_outliers()
        self.check_jumps()
        self.filtered_models = list(set(self.outliers) | set(self.jumps))
        return self.filtered_models


def drop_model(col_names, dict_or_df):
    """
    Drop columns with given names from either a dictionary of dataframes
    or a single dataframe.
    Parameters
    ----------
    col_names : list of str
        The list of model names to drop.
    dict_or_df : dict of pandas.DataFrame or pandas.DataFrame
        If a dict of dataframes, all dataframes in the dict will be edited.
        If a single dataframe, only that dataframe will be edited.
    Returns
    -------
    dict_of_dfs : dict of pandas.DataFrame or pandas.DataFrame
        The updated dictionary of dataframes or dataframe with dropped columns.
    """
    if isinstance(dict_or_df, dict):
        # loop through the dictionary and edit each dataframe
        for key in dict_or_df.keys():
            if all(col_name in dict_or_df[key].columns for col_name in col_names):
                dict_or_df[key] = dict_or_df[key].drop(columns=col_names)
        return dict_or_df
    elif isinstance(dict_or_df, pd.DataFrame):
        # edit the single dataframe
        if all(col_name in dict_or_df.columns for col_name in col_names):
            return dict_or_df.drop(columns=col_names)
    else:
        raise TypeError('Input must be a dictionary or a dataframe')


def read_era5l(file):
    """Reads ERA5-Land data, drops redundant columns, and adds DatetimeIndex.
    Resamples the dataframe to reduce the DatetimeIndex to daily resolution."""
    
    return pd.read_csv(file, **{
        'usecols':      ['temp', 'prec', 'dt'],
        'index_col':    'dt',
        'parse_dates':  ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})
    

def adjust_bias(predictand, predictor, method='normal_mapping'):
    """Applies bias correction to discrete periods individually."""
    # Read predictor data
    predictor = read_era5l(predictor)

    # Determine variable type based on the mean value
    var = 'temp' if predictand.mean().mean() > 100 else 'prec'

    # Adjust bias in discrete blocks as suggested by Switanek et al. (2017)
    correction_periods = [
        {'correction_range': ('1979-01-01', '2010-12-31'), 'extraction_range': ('1979-01-01', '1990-12-31')},
    ]
    for decade_start in range(1991, 2090, 10):
        correction_start = f"{decade_start - 10}-01-01"
        correction_end = f"{decade_start + 19}-12-31"
        extraction_start = f"{decade_start}-01-01"
        extraction_end = f"{decade_start + 9}-12-31"

        correction_periods.append({
            'correction_range': (correction_start, correction_end),
            'extraction_range': (extraction_start, extraction_end)
        })

    correction_periods.append({
        'correction_range': ('2081-01-01', '2100-12-31'),
        'extraction_range': ('2091-01-01', '2100-12-31')
    })

    # Store corrected periods
    corrected_data_list = []
    training_period = slice('1979-01-01', '2022-12-31')

    for period in tqdm(correction_periods, desc="Bias Correction"):
        correction_start, correction_end = period['correction_range']
        extraction_start, extraction_end = period['extraction_range']

        correction_slice = slice(correction_start, correction_end)
        extraction_slice = slice(extraction_start, extraction_end)

        data_corr = pd.DataFrame()
        for col in predictand.columns:
            x_train = predictand[col][training_period].squeeze()
            y_train = predictor[training_period][var].squeeze()
            x_predict = predictand[col][correction_slice].squeeze()
            bc_corr = BiasCorrection(y_train, x_train, x_predict)
            corrected_col = pd.DataFrame(bc_corr.correct(method=method))
            data_corr[col] = corrected_col.loc[extraction_slice]

        corrected_data_list.append(data_corr)

    corrected_data = pd.concat(corrected_data_list, axis=0)
    return corrected_data


def confidence_interval(df):
    """
    Calculate the mean and 95% confidence interval for each row in a dataframe.
    Parameters:
    -----------
        df (pandas.DataFrame): The input dataframe.
    Returns:
    --------
        pandas.DataFrame: A dataframe with the mean and confidence intervals for each row.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    count = df.count(axis=1)
    ci = 1.96 * std / np.sqrt(count)
    ci_lower = mean - ci
    ci_upper = mean + ci
    df_ci = pd.DataFrame({'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper})
    return df_ci


def dict_filter(dictionary, filter_string):
    """Returns a dict with all elements of the input dict that contain a filter string in their keys."""
    return {key.split('_')[0]: value for key, value in dictionary.items() if filter_string in key}


def replace_values(target_df, source_df, source_column):
    """
    Replaces values in the overlapping period in the target dataframe with values
    from the source dataframe using the specified source column.

    Args:
        target_df (pd.DataFrame): Target dataframe where values will be replaced.
        source_df (pd.DataFrame): Source dataframe from which values will be taken.
        source_column (str): Column name in the source dataframe to use for replacement.

    Returns:
        pd.DataFrame: The target dataframe with updated values.
    """

    # Identify overlapping period based on index (datetime)
    overlapping_period = target_df.index.intersection(source_df.index)


    if len(overlapping_period) == 0:
        raise ValueError("No overlapping period between the source and target dataframes.")

    # Ensure the source dataframe has the required column
    if source_column not in source_df.columns:
        raise ValueError(f"The source dataframe does not have a column named '{source_column}'")
    
    # Get the replacement values from the source columnAdd commentMore actions
    replacement_values = source_df.loc[overlapping_period, source_column]

    assert len(overlapping_period) == len(
        replacement_values), "Mismatch in lengths of overlapping period and replacement values."

    # Apply these values to all columns in the target DataFrame in the overlapping period
    target_df.loc[overlapping_period] = replacement_values.values[:, None]

    return target_df


def get_si(fast_results: str, to_csv: bool = False) -> pd.DataFrame:
    """
    Computes the sensitivity indices of a given FAST simulation results file.
    Parameters
    ----------
    fast_results : str
        The path of the FAST simulation results file.
    to_csv : bool, optional
        If True, the sensitivity indices are saved to a CSV file with the same
        name as fast_results, but with '_sensitivity_indices.csv' appended to
        the end (default is False).
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the sensitivity indices and parameter
        names.
    """
    if fast_results.endswith(".csv"):
        fast_results = fast_results[:-4]  # strip .csv
    results = spotpy.analyser.load_csv_results(fast_results)
    # Suppress prints
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        SI = spotpy.analyser.get_sensitivity_of_fast(results, print_to_console=False)
    parnames = spotpy.analyser.get_parameternames(results)
    sens = pd.DataFrame(SI)
    sens['param'] = parnames
    sens.set_index('param', inplace=True)
    if to_csv:
        sens.to_csv(os.path.basename(fast_results) + '_sensitivity_indices.csv', index=False)
    return sens


def create_scenario_dict(tas: dict, pr: dict, scenario_nums: list) -> dict:
    """
    Create a nested dictionary of scenarios and models from two dictionaries of pandas DataFrames.
    Parameters
    ----------
    tas : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate model mean daily temperature (K) time series.
    pr : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate models mean daily precipitation (mm/day) time series.
    scenario_nums : list
        A list of integers representing the scenario numbers to include in the resulting dictionary.
    Returns
    -------
    dict
        A nested dictionary where the top-level keys are scenario names (e.g. 'SSP2', 'SSP5') and the values are
        dictionaries containing climate models as keys and the corresponding pandas DataFrames as values.
        The DataFrames have three columns: 'TIMESTAMP', 'T2', and 'RRR', where 'TIMESTAMP'
        represents the time step, 'T2' represents the mean daily temperature (K), and 'RRR' represents the mean
        daily precipitation (mm/day).
    """
    scenarios = {}
    for s in scenario_nums:
        s = 'SSP' + str(s)
        scenarios[s] = {}
        for m in tas[s].columns:
            model = pd.DataFrame({'T2': tas[s][m],
                                  'RRR': pr[s][m]})
            model = model.reset_index()
            mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
            scenarios[s].update(mod_dict)
    return scenarios


class MatildaBulkProcessor:
    """
    A class to run multiple MATILDA simulations for different input scenarios and models in single or multi-processing
    mode and store the results in a dictionary.
    Attributes
    ----------
    scenarios : dict
        A dictionary with scenario names as keys and a dictionary of climate models as values.
    matilda_settings : dict
        A dictionary of MATILDA settings.
    matilda_parameters : dict
        A dictionary of MATILDA parameter values.
    Methods
    -------
    run_single_process():
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
    run_multi_process():
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
    matilda_headless(df, matilda_settings, matilda_parameters):
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
    """

    def __init__(self, scenarios, matilda_settings, matilda_parameters):
        """
        Parameters
        ----------
        scenarios : dict
            A dictionary with scenario names as keys and a dictionary of models as values.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        """

        self.scenarios = scenarios
        self.matilda_settings = matilda_settings
        self.matilda_parameters = matilda_parameters

    @staticmethod
    def matilda_headless(df, matilda_settings, matilda_parameters):
        """
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe for the MATILDA simulation.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        Returns
        -------
        dict
            A dictionary containing the MATILDA model output and glacier rescaling factor.
        """

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                output = matilda_simulation(df, **matilda_settings, **matilda_parameters)
        return {'model_output': output[0], 'glacier_rescaling': output[5]}

    def run_single_process(self):
        """
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        # Loop over the scenarios with progress bar
        for scenario in self.scenarios.keys():
            model_dict = {}  # Create an empty dictionary to store the model outputs
            # Loop over the models with progress bar
            for model in tqdm(self.scenarios[scenario].keys(), desc=scenario):
                # Get the dataframe for the current scenario and model
                df = self.scenarios[scenario][model]
                # Run the model simulation and get the output while suppressing prints
                model_output = self.matilda_headless(df, self.matilda_settings, self.matilda_parameters)
                # Store the list of output in the model dictionary
                model_dict[model] = model_output
            # Store the model dictionary in the scenario dictionary
            out_dict[scenario] = model_dict
        return out_dict

    def run_multi_process(self, num_cores=2):
        """
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        with Pool(num_cores) as pool:
            # Loop over the scenarios with progress bar
            for scenario in tqdm(self.scenarios.keys(), desc="Scenarios SSP2 and SSP5"):
                model_dict = {}  # Create an empty dictionary to store the model outputs
                # Loop over the models with progress bar
                model_list = [self.scenarios[scenario][m] for m in self.scenarios[scenario].keys()]
                for model, model_output in zip(self.scenarios[scenario], pool.map(
                        partial(self.matilda_headless, matilda_settings=self.matilda_settings,
                                matilda_parameters=self.matilda_parameters), model_list)):
                    model_dict[model] = model_output
                # Store the model dictionary in the scenario dictionary
                out_dict[scenario] = model_dict
            pool.close()

        return out_dict

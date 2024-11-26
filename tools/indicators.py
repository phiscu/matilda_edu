from tools.helpers import hydrologicalize
import spotpy.hydrology.signatures as sig
from climate_indices.indices import spei, spi, Distribution
from climate_indices import compute, utils
import pandas as pd
import numpy as np
import inspect


def prec_minmax(df):
    """
    Compute the months of extreme precipitation for each year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily precipitation data with a datetime index and a 'total_precipitation' column.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the months of extreme precipitation as a number for every calendar year.
    """
    # Use water years
    df = hydrologicalize(df)
    # group the data by year and month and sum the precipitation values
    grouped = df.groupby([df.water_year, df.index.month]).sum()
    # get the month with extreme precipitation for each year
    max_month = grouped.groupby(level=0)['total_precipitation'].idxmax()
    min_month = grouped.groupby(level=0)['total_precipitation'].idxmin()
    max_month = [p[1] for p in max_month]
    min_month = [p[1] for p in min_month]
    # create a new dataframe
    result = pd.DataFrame({'max_prec_month': max_month, 'min_prec_month': min_month},
                          index=pd.to_datetime(df.water_year.unique(), format='%Y'))
    return result


# Day of the Year with maximum flow
def peak_doy(df, smoothing_window_peakdoy=7):
    """
    Compute the day of the calendar year with the peak value for each hydrological year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily data with a datetime index.
    smoothing_window_peakdoy : int, optional
        The window size of the rolling mean used for smoothing the data.
        Default is 7.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the day of the year with the peak value for each hydrological year.
    """
    # Use water years
    df = hydrologicalize(df)

    # find peak day for each hydrological year
    peak_dates = []
    for year in df.water_year.unique():
        # slice data for hydrological year
        hy_data = df.loc[df.water_year == year, 'total_runoff']

        # smooth data using rolling mean with window of 7 days
        smoothed_data = hy_data.rolling(smoothing_window_peakdoy, center=True).mean()

        # find day of peak value
        peak_day = smoothed_data.idxmax().strftime('%j')

        # append peak day to list
        peak_dates.append(peak_day)

    # create output dataframe with DatetimeIndex
    output_df = pd.DataFrame({'Hydrological Year': df.water_year.unique(),
                              'peak_day': pd.to_numeric(peak_dates)})
    output_df.index = pd.to_datetime(output_df['Hydrological Year'], format='%Y')
    output_df = output_df.drop('Hydrological Year', axis=1)

    return output_df


# Melting season
def melting_season(df, smoothing_window_meltseas=14, min_weeks=10):
    """
    Compute the start, end, and length of the melting season for each calendar year based on the daily
    the rolling mean of the temperature.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily mean temperature data with a datetime index.

    smoothing_window_meltseas : int, optional
        The size of the rolling window in days used to smooth the temperature data. Default is 14.

    min_weeks : int, optional
        The minimum number of weeks that the melting season must last for it to be considered valid. Default is 10.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the start, end, and length of the melting season for each calendar year, with a DatetimeIndex.
    """

    # Compute rolling mean of temperature data
    rolling_mean = df['avg_temp_catchment'].rolling(window=smoothing_window_meltseas).mean()

    # Find start of melting season for each year (first day above 0°C)
    start_mask = rolling_mean > 0
    start_mask = start_mask.groupby(df.index.year).apply(lambda x: x.index[np.nanargmax(x)])
    start_dates = start_mask - pd.Timedelta(days=smoothing_window_meltseas - 1)      # rolling() selects last day, we want the first

    # Add minimum length of melting season to start dates
    earliest_end_dates = start_dates + pd.Timedelta(weeks=min_weeks)

    # group rolling_mean by year and apply boolean indexing to replace values before start_dates with 999 in every year
    rolling_mean = rolling_mean.groupby(rolling_mean.index.year).\
        apply(lambda x: np.where(x.index < earliest_end_dates.loc[x.index.year], 999, x))

    # Transform the resulting array back to a time series with DatetimeIndex
    rolling_mean = pd.Series(rolling_mean.values.flatten())
    rolling_mean = rolling_mean.explode()
    rolling_mean = pd.DataFrame({'rolling_mean': rolling_mean}).set_index(df.index)

    # Filter values below 0 (including 999!)
    end_mask = rolling_mean < 0
    # Find end of melting season
    end_mask = end_mask.groupby(df.index.year).apply(lambda x: x.index[np.nanargmax(x)])
    end_dates = end_mask - pd.Timedelta(days=smoothing_window_meltseas - 1)

    # Compute length of melting season for each year
    lengths = (end_dates - start_dates).dt.days

    # Assemble output dataframe
    output_df = pd.DataFrame({'melt_season_start': [d.timetuple().tm_yday for d in start_dates],
                              'melt_season_end': [d.timetuple().tm_yday for d in end_dates]},
                             index=pd.to_datetime(df.index.year.unique(), format='%Y'))
    output_df['melt_season_length'] = output_df.melt_season_end - output_df.melt_season_start

    return output_df


# Aridity
def aridity(df, hist_starty=1986, hist_endy=2015):
    """
    Calculates aridity indexes from precipitation, and potential and actual evaporation respectively. Aridity is defined
    as mean annual ratio of potential/actual evapotranspiration and precipitation. The indexes are defined as the
    relative change of a 30 years period compared to a given historical period. Uses hydrological years (Oct - Sep).
    Inspired by climateinformation.org (https://doi.org/10.5194/egusphere-egu23-16216).
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers', 'actual_evaporation', and 'prec_off_glaciers'.
    hist_starty : int, optional
        Start year of the historical period in YYYY format. Default is 1986.
    hist_endy : int, optional
        End year of the historical period in YYYY format. Default is 2015.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the relative change in aridity over time.
        Columns:
            - 'actual_aridity': Relative change in actual aridity.
            - 'potential_aridity': Relative change in potential aridity.
    """
    # Use water years
    df = hydrologicalize(df)
    # Potential evapotranspiration (PET)
    pet = df['evap_off_glaciers']
    # Actual evapotranspiration (AET)
    aet = df['actual_evaporation']
    # Precipitation
    prec = df['prec_off_glaciers']
    # Calculate the potential aridity as ratio of AET/PET to precipitation
    aridity_pot = pet.groupby(df['water_year']).sum() / prec.groupby(df['water_year']).sum()
    aridity_act = aet.groupby(df['water_year']).sum() / prec.groupby(df['water_year']).sum()
    # Filter historical period
    hist_pot = aridity_pot[(aridity_pot.index >= hist_starty) & (aridity_pot.index <= hist_endy)].mean()
    hist_act = aridity_act[(aridity_act.index >= hist_starty) & (aridity_act.index <= hist_endy)].mean()
    # Calculate rolling mean with a 30y period
    aridity_pot_rolling = aridity_pot.rolling(window=30).mean()
    aridity_act_rolling = aridity_act.rolling(window=30).mean()
    # Calculate the relative change in the aridity indexes
    pot_rel = 100 * (aridity_pot_rolling - hist_pot) / hist_pot
    act_rel = 100 * (aridity_act_rolling - hist_act) / hist_act
    # Concat in one dataframe
    aridity = pd.DataFrame({'actual_aridity': act_rel, 'potential_aridity': pot_rel})
    aridity.set_index(pd.to_datetime(df.water_year.unique(), format='%Y'), inplace=True)
    aridity = aridity.dropna()


    return aridity


# Dry spells
def dry_spells(df, dry_spell_length=5):
    """
    Compute the total length of dry spells in days per year. A dry spell is defined as a period for which the rolling
    mean of evaporation in a given window exceeds precipitation. Uses hydrological years (Oct - Sep).
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers' and 'prec_off_glaciers' with daily evaporation and
        precipitation data, respectively.
    dry_spell_length : int, optional
        Length of the rolling window in days. Default is 30.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the number of days for which the rolling mean of evaporation exceeds precipitation for each
        year in the input DataFrame.
    """
    # Use hydrological years
    df = hydrologicalize(df)
    # Find number of days when the rolling mean of evaporation exceeds precipitation
    periods = []
    for year in df.water_year.unique():
        year_data = df.loc[df.water_year == year]
        evap_roll = year_data['evap_off_glaciers'].rolling(window=dry_spell_length).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=dry_spell_length).mean()

        dry = evap_roll[evap_roll - prec_roll > 0]
        periods.append(len(dry))

    # Assemble the output dataframe
    output_df = pd.DataFrame(
        {'dry_spell_days': periods},
        index=pd.to_datetime(df.water_year.unique(), format='%Y'))

    return output_df


# Hydrological signatures
def get_qhf(data, global_median, measurements_per_day=1):
    """
    Variation of spotpy.hydrology.signatures.get_qhf() that allows definition of a global
    median to investigate long-term trends.
    Calculates the frequency of high flow events defined as :math:`Q > 9 \\cdot Q_{50}`
    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math: :math:`yr^{-1}`
    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: Q_{HF}, Q_{HD}
    """

    def highflow(value, median):
        return value > 9 * median

    fq, md = sig.flow_event(data, highflow, global_median)

    return fq * measurements_per_day * 365, md / measurements_per_day


def get_qlf(data, global_mean, measurements_per_day=1):
    """
    Variation of spotpy.hydrology.signatures.get_qlf() that allows comparison of
    individual years with a global mean to investigate long-term trends.
    Calculates the frequency of low flow events defined as
    :math:`Q < 0.2 \\cdot \\overline{Q_{mean}}`
    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given
    in :math:`yr^{-1}` and for the whole timeseries
    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: Q_{LF}, Q_{LD}
    """

    def lowflow(value, mean):
        return value < 0.2 * mean

    fq, md = sig.flow_event(data, lowflow, global_mean)
    return fq * measurements_per_day * 365, md / measurements_per_day


def hydrological_signatures(df):
    """
    Calculate hydrological signatures for a given input dataframe.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a column 'total_runoff' and a DatetimeIndex.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated hydrological signatures for each year in the input dataframe.
        The columns of the output dataframe are as follows:
         - 'q5': the 5th percentile of total runoff for each year
        - 'q50': the 50th percentile of total runoff for each year
        - 'q95': the 95th percentile of total runoff for each year
        - 'qlf_freq': the frequency of low flow events (defined as Q < 2*Qmean_global) for each year, in yr^⁻1
        - 'qlf_dur': the mean duration of low flow events (defined as Q < 2*Qmean_global) for each year, in days
        - 'qhf_freq': the frequency of high flow events (defined as Q > 9*Q50_global) for each year, in yr^⁻1
        - 'qhf_dur': the mean duration of high flow events (defined as Q > 9*Q50_global) for each year, in days
    """
    # Create lists of quantile functions to apply and column names
    functions = [sig.get_q5, sig.get_q50, sig.get_q95]
    cols = ['q5', 'q50', 'q95']

    # Create an empty dataframe to store the results
    results_df = pd.DataFrame()

    # Use water_year
    df = hydrologicalize(df)

    # Loop through each year in the input dataframe
    for year in df.water_year.unique():
        # Select the data for the current year
        year_data = df[df.water_year == year].total_runoff
        # Apply each quantile function to the year data and store the results in a dictionary
        year_results = {}
        for i, func in enumerate(functions):
            year_results[cols[i]] = func(year_data)
        # Calculate frequency and duration of global low flows
        qlf_freq, qlf_dur = get_qlf(year_data, np.mean(df.total_runoff))
        year_results['qlf_freq'] = qlf_freq
        year_results['qlf_dur'] = qlf_dur
        # Calculate frequency and duration of global high flows
        qhf_freq, qhf_dur = get_qhf(year_data, np.median(df.total_runoff))
        year_results['qhf_freq'] = qhf_freq
        year_results['qhf_dur'] = qhf_dur
        # Convert the dictionary to a dataframe and append it to the results dataframe
        year_results_df = pd.DataFrame(year_results, index=[year])
        results_df = pd.concat([results_df, year_results_df])

    results_df.set_index(pd.to_datetime(df.water_year.unique(), format='%Y'), inplace=True)

    return results_df


# Drought indicators
def drought_indicators(df, freq='ME', dist='gamma'):
    """
    Calculate the climatic water balance, SPI (Standardized Precipitation Index), and
    SPEI (Standardized Precipitation Evapotranspiration Index) for 1, 3, 6, 12, and 24 months.
    Parameters
    ----------
    df : pandas.DataFrame
         Input DataFrame containing columns 'prec_off_glaciers' and 'evap_off_glaciers'.
    freq : str, optional
         Resampling frequency for precipitation and evaporation data. Default is 'M' for monthly.
    dist : str, optional
         Distribution for SPI and SPEI calculation. Either Pearson-Type III ('pearson') or
         Gamma distribution ('gamma'). Default is 'gamma'.
    Returns
    -------
    pandas.DataFrame
         DataFrame containing the calculated indicators: 'clim_water_balance', 'spi', and 'spei'.
         Index is based on the resampled frequency of the input DataFrame.
    Raises
    ------
    ValueError
         If 'freq' is not 'D' or 'ME'.
         If 'dist' is not 'pearson' or 'gamma'.
    Notes
    -----
    SPI (Standardized Precipitation Index) and SPEI (Standardized Precipitation Evapotranspiration Index)
    are drought indicators that are used to quantify drought severity and duration.
    'clim_water_balance' is the difference between total precipitation and total evapotranspiration.
    If 'freq' is 'D', the input data is transformed from Gregorian to a 366-day format for SPI and SPEI calculation,
    and then transformed back to Gregorian format for output.
    The default distribution for SPI and SPEI calculation is Gamma.
    The calibration period for SPI and SPEI calculation is from 1981 to 2020.
    """
    # Check if frequency is valid
    if freq != 'D' and freq != 'ME':
        raise ValueError("Invalid value for 'freq'. Choose either 'D' or 'ME'.")

    # Resample precipitation and evaporation data based on frequency
    prec = df.prec_off_glaciers.resample(freq).sum().values
    evap = df.evap_off_glaciers.resample(freq).sum().values

    # Calculate water balance
    water_balance = prec - evap

    # If frequency is daily, transform data to 366-day format
    if freq == 'D':
        prec = utils.transform_to_366day(prec, year_start=df.index.year[0],
                                         total_years=len(df.index.year.unique()))
        evap = utils.transform_to_366day(evap, year_start=df.index.year[0],
                                         total_years=len(df.index.year.unique()))

    # Set distribution based on input
    if dist == 'pearson':
        distribution = Distribution.pearson
    elif dist == 'gamma':
        distribution = Distribution.gamma
    else:
        raise ValueError("Invalid value for 'dist'. Choose either 'pearson' or 'gamma'.")

    # Set periodicity based on frequency
    if freq == 'D':
        periodicity = compute.Periodicity.daily
    elif freq == 'ME':
        periodicity = compute.Periodicity.monthly

    # Set common parameters
    common_params = {'distribution': distribution,
                     'periodicity': periodicity,
                     'data_start_year': 1981,
                     'calibration_year_initial': 1981,
                     'calibration_year_final': 2020}

    # Set parameters for SPEI calculation
    spei_params = {'precips_mm': prec,
                   'pet_mm': evap,
                   **common_params}

    # Set parameters for SPI calculation
    spi_params = {'values': prec,
                  **common_params}

    # Calculate SPI and SPEI for various periods
    drought_df = pd.DataFrame()
    for s in [1, 3, 6, 12, 24]:
        spi_arr = spi(**spi_params, scale=s)
        spei_arr = spei(**spei_params, scale=s)
        # If frequency is daily, transform data back to Gregorian format
        if freq == 'D':
            spi_arr = utils.transform_to_gregorian(spi_arr, df.index.year[0])
            spei_arr = utils.transform_to_gregorian(spei_arr, df.index.year[0])
        drought_df['spi' + str(s)] = spi_arr
        drought_df['spei' + str(s)] = spei_arr
    drought_df.set_index(df.resample(freq).mean().index, inplace=True)

    # DataFrame resample Dataframe
    out_df = pd.DataFrame({'clim_water_balance': water_balance}, index=df.resample(freq).sum().index)
    out_df = pd.concat([out_df.resample('YS').sum(), drought_df.resample('YS').mean()], axis=1).rename(lambda x: x.replace(day=1))

    return out_df


# Wrapper function

def cc_indicators(df, **kwargs):
    """
    Apply a list of climate change indicator functions to output DataFrame of MATILDA and concatenate
    the output columns into a single DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    **kwargs : optional
        Optional arguments to be passed to the functions in the list. Possible arguments are 'smoothing_window_peakdoy',
        'smoothing_window_meltseas', 'min_weeks', and 'dry_spell_length'.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output columns of all functions applied to the input DataFrame.
    Notes
    -----
    The list of functions to apply is hard-coded into the function and cannot be modified from outside.
     The optional arguments are passed to the respective functions only if they are relevant for the respective
     function.
     If no optional arguments are passed, the function is applied to the input DataFrame with default arguments.
    """
    # List of all functions to apply
    functions = [prec_minmax, peak_doy, melting_season, aridity, dry_spells, hydrological_signatures, drought_indicators]
    # Empty result dataframe
    indicator_df = pd.DataFrame()
    # Loop through all functions
    for func in functions:
        func_kwargs = {}
        # Apply only those optional kwargs relevant for the respective function
        for kwarg in kwargs:
            if kwarg in inspect.getfullargspec(func)[0]:
                func_kwargs.update({kwarg: kwargs.get(kwarg)})
        result = func(df, **func_kwargs)
        # Concat all output columns in one dataframe
        indicator_df = pd.concat([indicator_df, result], axis=1)

    return indicator_df


indicator_vars = {
 'max_prec_month': ('Month with Maximum Precipitation', '-'),
 'min_prec_month': ('Month with Minimum Precipitation', '-'),
 'peak_day': ('Timing of Peak Runoff', 'DoY'),
 'melt_season_start': ('Beginning of Melting Season', 'DoY'),
 'melt_season_end': ('End of Melting Season', 'DoY'),
 'melt_season_length': ('Length of Melting Season', 'd'),
 'actual_aridity': ('Relative Change of Actual Aridity', '%'),
 'potential_aridity': ('Relative Change of Potential Aridity', '%'),
 'dry_spell_days': ('Total Length of Dry Spells per year', 'd/a'),
 'qlf_freq': ('Frequency of Low-flow events', 'yr^-1'),
 'qlf_dur': ('Mean Duration of Low-flow events', 'd'),
 'qhf_freq': ('Frequency of High-flow events', 'yr^-1'),
 'qhf_dur': ('Mean Duration of High-flow events', 'd'),
 'clim_water_balance': ('Climatic Water Balance', 'mm w.e.'),
 'spi1': ('Standardized Precipitation Index (1 month)', '-'),
 'spei1': ('Standardized Precipitation Evaporation Index (1 month)', '-'),
 'spi3': ('Standardized Precipitation Index (3 months)', '-'),
 'spei3': ('Standardized Precipitation Evaporation Index (3 months)', '-'),
 'spi6': ('Standardized Precipitation Index (6 months)', '-'),
 'spei6': ('Standardized Precipitation Evaporation Index (6 months)', '-'),
 'spi12': ('Standardized Precipitation Index (12 months)', '-'),
 'spei12': ('Standardized Precipitation Evaporation Index (12 months)', '-'),
 'spi24': ('Standardized Precipitation Index (24 months)', '-'),
 'spei24': ('Standardized Precipitation Evaporation Index (24 months)', '-')
}
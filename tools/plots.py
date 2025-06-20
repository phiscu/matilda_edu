import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import pandas as pd
import warnings
from matplotlib.legend import Legend
import probscale
from tools.helpers import read_era5l
import configparser
from scipy import stats
import numpy as np
from tools.helpers import parquet_to_dict, read_yaml, pickle_to_dict
import warnings
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib as mpl
from dash import Dash, dcc, html, Input, Output
from jupyter_server import serverapp


# Use Seaborn white style
sns.set_style("white")

# Use LaTeX for rendering text
plt.rcParams['text.usetex'] = True     # overwrites several font settings

# Set consistent font type, size, and weight
#plt.rcParams['font.size'] = 14
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.weight'] = 'bold'

# Set figure and subplot title size and weight
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['figure.titleweight'] = 'heavy'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'

# Set consistent grid style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linewidth'] = 0.5

# Set axis style
plt.rcParams['axes.labelsize']= 15
#plt.rcParams['axes.linewidth'] = 1.0

# get style for matplotlib plots
# plt_style = ast.literal_eval(config['CONFIG']['PLOT_STYLE'])

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",            # or 'sans-serif', 'monospace', etc.
    "font.serif": ["Computer Modern"], # default LaTeX serif font
})



def df2long(df, intv_sum='ME', intv_mean='YE', precip=False):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""

    if precip:
        df = df.resample(intv_sum).sum().resample(intv_mean, label='left').mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='prec')
    else:
        df = df.resample(intv_mean, label='left').mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='temp')
    return df


def cmip_plot(ax, df, target, title=None, precip=False, smooth_window=10, agg_level='monthly',
              target_label='Target', show_target_label=False):
    """Plots climate model and target data using moving window smoothing."""
    
    try:
        smooth_window = float(smooth_window)
    except ValueError:
        raise TypeError(f"smooth_window must be a number, got {type(smooth_window).__name__}")

    if precip:
        # Define aggregation frequency based on agg_level
        if agg_level == 'monthly':
            resample_freq = 'ME'
            smooth_window_units = int(smooth_window * 12)  # Convert years to months
        elif agg_level == 'annual':
            resample_freq = 'YE'
            smooth_window_units = int(smooth_window)  # Smooth window directly in years
        else:
            raise ValueError(f"Invalid agg_level: {agg_level}. Use 'monthly' or 'annual'.")

        # Convert daily data to specified aggregation level (monthly or annual)
        df_agg = df.resample(resample_freq).sum()
        target_agg = target['prec'].resample(resample_freq).sum()

        # Apply rolling mean on aggregated data
        ax.plot(df_agg.rolling(window=smooth_window_units, center=True).mean().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target_agg.rolling(window=smooth_window_units, center=True).mean(), 
                            linewidth=1.2, c='black', label=target_label)
    else:
        # Convert smoothing window from years to days for temperature
        smooth_window_days = int(smooth_window * 365.25)  # Account for leap years
        
        # Apply rolling mean on daily data
        ax.plot(df.rolling(window=smooth_window_days, center=True).mean().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target['temp'].rolling(window=smooth_window_days, center=True).mean(), 
                            linewidth=1.2, c='black', label=target_label)

    ax.margins(x=0.001)

    if show_target_label:
        ax.legend(handles=[era_plot], loc='upper left')

    if title:
        ax.set_title(title)

    # Set y-labels for left-side plots using the corrected column check
    if ax.get_subplotspec().colspan.start == 0:
        if precip:
            ylabel = '[mm]'
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('[K]')

    ax.grid(True)


def cmip_plot_combined(data, target, title=None, precip=False, agg_level='monthly', smooth_window=10,
                       target_label='Target', show=False, fig_path=None):
    """Combines multiple subplots of climate data in different scenarios before and after bias adjustment.
    Uses moving window smoothing in years instead of days."""
    
    figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
    
    t_kwargs = {'target': target, 'smooth_window': smooth_window, 'target_label': target_label}
    p_kwargs = {'target': target, 'smooth_window': smooth_window, 'target_label': target_label,
                'precip': True, 'agg_level': agg_level}

    if not precip:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **t_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **t_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **t_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **t_kwargs)
        if title:
            figure.suptitle(f'{smooth_window} Year Rolling Mean of {agg_level.capitalize()} Precipitation')
    else:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **p_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **p_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **p_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **p_kwargs)
        if title:
            figure.suptitle(f'{smooth_window} Year Rolling Mean of Air Temperature')

    figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.15, top=0.92)

    if fig_path is not None:
        plt.savefig(fig_path)
    if show:
        plt.show()

            

def vplots(before, after, target, target_label='Target', precip=False, show=False, fig_path=None):
    """Creates violin plots of the kernel density estimation for all models before and after bias adjustment."""

    period = slice('1979-01-01', '2022-12-31')
    if precip:
        var = 'prec'
        var_label = 'Annual Precipitation'
        unit = ' [mm]'
    else:
        var = 'temp'
        var_label = 'Mean Annual Air Temperature'
        unit = ' [K]'

    for i in before.keys():
        before[i] = before[i].loc[period].copy()
        before[i][target_label] = target[var][period]

    for i in after.keys():
        after[i] = after[i].loc[period].copy()
        after[i][target_label] = target[var][period]

    fig = plt.figure(figsize=(20, 20))
    outer = fig.add_gridspec(1, 2)
    inner = outer[0].subgridspec(2, 1)
    axis = inner.subplots(sharex='col')


    all_data = pd.concat([df2long(before[i], precip=precip, intv_sum='YE') for i in before.keys()] +
                         [df2long(after[i], precip=precip, intv_sum='YE') for i in after.keys()])
    xmin, xmax = all_data[var].min(), all_data[var].max()

    xlim = (xmin * 0.95, xmax * 1.05) if precip else (xmin - 0.5, xmax + 0.5)

    # ---------------- "Before" Plots ----------------
    for (i, k) in zip(before.keys(), range(0, 4, 1)):
        df = df2long(before[i], precip=precip, intv_sum='YE')
        axis[k].grid()

        sns.violinplot(ax=axis[k], x=var, y='model', hue='model', data=df,
                       density_norm="count", bw_adjust=.2, palette='husl', legend=False)
        
        # Set y-axis labels manually
        axis[k].set_yticks(range(len(df['model'].unique())))
        axis[k].set_yticklabels(df['model'].unique())
        axis[k].set(xlim=xlim)
        axis[k].set_ylabel(i, fontsize=20, fontweight='bold')

        # Separate Reanalysis data
        tick_number = len(df['model'].unique())
        before_last = tick_number - 1.5
        axis[k].axhline(before_last, color='black', linestyle='--', linewidth=1)

        if k == 0:
            axis[k].set_title('Before Scaled Distribution Mapping')        

    plt.xlabel(var_label + unit)

    # ---------------- "After" Plots ----------------
    inner = outer[1].subgridspec(2, 1)
    axis = inner.subplots(sharex='col')

    for (i, k) in zip(after.keys(), range(0, 4, 1)):
        df = df2long(after[i], precip=precip, intv_sum='YE')
        axis[k].grid()

        sns.violinplot(ax=axis[k], x=var, y='model', hue='model', data=df,
                       density_norm="count", bw_adjust=.2, palette='husl', legend=False)

        axis[k].set(xlim=xlim)
        axis[k].set_ylabel(i)
        axis[k].get_yaxis().set_visible(False)

        # Separate Reanalysis data
        tick_number = len(df['model'].unique())
        before_last = tick_number - 1.5
        axis[k].axhline(before_last, color='black', linestyle='--', linewidth=1)

        if k == 0:
            axis[k].set_title('After Scaled Distribution Mapping')

    plt.xlabel(var_label + unit)

    # ---------------- Global Title ----------------
    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    fig.suptitle('Kernel Density Estimation of ' + var_label + ' (' + starty + '-' + endy + ')')

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.subplots_adjust(hspace=0.05)


    if fig_path is not None:
        plt.savefig(fig_path)

    if show:
        plt.show()
        

def cmip_plot_ensemble(cmip, target, precip=False, intv_sum='ME', intv_mean='YE', figsize=(8, 6), show=True, fig_path=None):
    """
    Plots the multi-model mean of climate scenarios including the 90% confidence interval.
    Parameters
    ----------
    cmip: dict
        A dictionary with keys representing the different CMIP6 models and scenarios as pandas dataframes
        containing data of temperature and/or precipitation.
    target: pandas.DataFrame
        Dataframe containing the historical reanalysis data.
    precip: bool
        If True, plot the mean precipitation. If False, plot the mean temperature. Default is False.
    intv_sum: str
        Interval for precipitation sums. Default is monthly ('ME').
    intv_mean: str
        Interval for the mean of temperature data or precipitation sums. Default is annual ('YE').
    figsize: tuple
        Figure size for the plot. Default is (8,6).
    show: bool
        If True, show the resulting plot. If False, do not show it. Default is True.
    """

    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)

    # Define color palette
    colors = ['darkorange', 'orange', 'darkblue', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(cmip.keys(), colors)}
    
    #figure.tight_layout(rect=[0, 0.02, 1, 0.95])  # Make some room at the bottom

    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='TIMESTAMP', y='prec', color=col_dict[i])
        axis.set(xlabel=None, ylabel='[mm]')
        if intv_sum=='ME':
            figure.suptitle('Mean Monthly Precipitation')
        elif intv_sum=='YE':
            figure.suptitle('Mean Annual Precipitation')
        target_plot = axis.plot(target.resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5, c='black',
                             label='ERA5', linestyle='dashed')
    else:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_mean=intv_mean)
            sns.lineplot(data=df, x='TIMESTAMP', y='temp', color=col_dict[i])
        axis.set(xlabel=None, ylabel='[K]')
        if intv_mean=='10YE':
            figure.suptitle('Mean 10y Air Temperature')
        elif intv_mean == 'YE':
            figure.suptitle('Mean Annual Air Temperature')
        elif intv_mean == 'ME':
            figure.suptitle('Mean Monthly Air Temperature')
        target_plot = axis.plot(target.resample(intv_mean).mean(), linewidth=1.5, c='black',
                         label='ERA5', linestyle='dashed')
    axis.legend(['SSP2 raw', '_ci1', 'SSP2 adjusted', '_ci2', 'SSP5 raw', '_ci3', 'SSP5 adjusted', '_ci4'],
                loc="upper center", bbox_to_anchor=(0.39, -0.14), ncol=4,
                frameon=False)  # First legend --> Workaround as seaborn lists CIs in legend
    leg = Legend(axis, target_plot, ['ERA5L'], loc='upper center', bbox_to_anchor=(0.89, -0.14), ncol=1,
                 frameon=False)  # Second legend (ERA5)
    axis.add_artist(leg)
    axis.grid(True, linestyle='--', alpha=0.7)  # Dashed lines with transparency
    axis.margins(x=0.001)
    
    if fig_path is not None:
        plt.savefig(fig_path)
    if show:
        plt.show()
    warnings.filterwarnings(action='always')


def prob_plot(original, target, corrected, ax, title=None, ylabel="Temperature [K]", **kwargs):
    """
    Combines probability plots of climate model data before and after bias adjustment
    and the target data.

    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    ax : matplotlib.axes.Axes
        The axes on which to plot the probability plot.
    title : str, optional
        The title of the plot. Default is None.
    ylabel : str, optional
        The label for the y-axis. Default is "Temperature [K]".
    **kwargs : dict, optional
        Additional keyword arguments passed to the probscale.probplot() function.

    Returns
    -------
    fig : matplotlib Figure
        The generated figure.
    """

    scatter_kws = dict(label="", marker=None, linestyle="-")
    common_opts = dict(plottype="qq", problabel="", datalabel="", **kwargs)

    scatter_kws["label"] = "original"
    fig = probscale.probplot(original, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "target"
    fig = probscale.probplot(target, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "adjusted"
    fig = probscale.probplot(corrected, ax=ax, scatter_kws=scatter_kws, **common_opts)

    ax.set_title(title)

    ax.set_xlabel("Standard Normal Quantiles")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    score = round(target.corr(corrected), 2)
    ax.text(0.05, 0.8, f"R² = {score}", transform=ax.transAxes, fontsize=15)

    return fig


def pp_matrix(original, target, corrected, scenario=None, nrow=7, ncol=5, precip=False, show=False, fig_path=None):
    """
    Arranges the prob_plots of all CMIP6 models in a matrix and adds the R² score.

    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    scenario : str, optional
        The climate scenario to be added to the plot title.
    nrow : int, optional
        The number of rows in the plot matrix. Default is 7.
    ncol : int, optional
        The number of columns in the plot matrix. Default is 5.
    precip : bool, optional
        Indicates whether the data is precipitation data. Default is False.
    show : bool, optional
        Indicates whether to display the plot. Default is False.

    Returns
    -------
    None
    """

    period = slice('1979-01-01', '2022-12-31')
    if precip:
        var = 'Precipitation'
        var_label = 'Monthly ' + var
        unit = ' [mm]'
        original = original.resample('ME').sum()
        target = target.resample('ME').sum()
        corrected = corrected.resample('ME').sum()
    else:
        var = 'Temperature'
        var_label = 'Daily Mean ' + var
        unit = ' [K]'

    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate(original.columns):
        ax = plt.subplot(nrow, ncol, i + 1)
        prob_plot(original[col][period], target[period],
                  corrected[col][period], ax=ax, ylabel=var + unit)
        ax.set_title(col, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['original (CMIP6 raw)', 'target (ERA5-Land)', 'adjusted (CMIP6 after SDM)'], loc='lower right',
               bbox_to_anchor=(0.96, 0.024), fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    if scenario is None:
        fig.suptitle('Probability Plots of CMIP6 and ERA5-Land ' + var_label + ' (' + starty + '-' + endy + ')')
    else:
        fig.suptitle('Probability Plots of CMIP6 (' + scenario + ') and ERA5-Land ' + var_label +
                     ' (' + starty + '-' + endy + ')')
    plt.subplots_adjust(top=0.93)
    if fig_path is not None:
        plt.savefig(fig_path)
    if show:
        plt.show()


class MatildaSummary:
    def __init__(self, dir_input, dir_output, settings):
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.settings = settings
        self.arrow_props = dict(facecolor='grey', edgecolor='grey', arrowstyle='-', linewidth=0.5)
        self.matilda_scenarios = None
        self.obs = None
        self.df_era5 = None
        self.tas = None
        self.pr = None
        self.runoff = None
        self.evaporation = None
        self.precipitation = None
        self.glacier_area = None
        self.snow_melt = None
        self.ice_melt = None
        self.off_melt = None
        self.melt_ssp2 = None
        self.melt_ssp5 = None
        self.melt_diff = None

    def load_data(self):
        """Load all necessary data for plotting"""
        # Load observation data
        self.obs = pd.read_csv(self.dir_input + 'obs_runoff_example.csv')
        self.obs.index = pd.to_datetime(self.obs['Date'])
        self.obs['Date'] = self.obs.index
        self.obs["Qobs"] = self.obs["Qobs"] * 86400 / (self.settings['area_cat'] * 1000000) * 1000

        # Load ERA5 data
        self.df_era5 = pd.read_csv(self.dir_output + 'ERA5L.csv', **{
            'usecols': ['temp', 'prec', 'dt'],
            'index_col': 'dt',
            'parse_dates': ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})

        # Adjust start date
        self.df_era5.index = pd.to_datetime(self.df_era5.index)
        self.df_era5 = self.df_era5[self.df_era5.index >= '2000-01-01']
        self.obs = self.obs[self.obs.index >= '2000-01-01']

        # Load climate model data
        self.tas = pickle_to_dict(f"{self.dir_output}cmip6/adjusted/tas.pickle")
        self.pr = pickle_to_dict(f"{self.dir_output}cmip6/adjusted/pr.pickle")
        self.adjust_startdate(self.tas)
        self.adjust_startdate(self.pr)

        # Load MATILDA scenarios
        self.matilda_scenarios = pickle_to_dict(f"{self.dir_output}cmip6/adjusted/matilda_scenarios.pickle")
        
        # Prepare data for plotting
        self.prepare_data_for_plot()

    def adjust_startdate(self, data_dict, start_date='2000-01-01'):
        """Adjust start date for dictionaries of dataframes"""
        for key in data_dict:
            df = data_dict[key]
            df.index = pd.to_datetime(df.index)
            data_dict[key] = df[df.index >= start_date]

    def get_matilda_result_all_models(self, scenario, result_name, dict_name='model_output'):
        """Extract specific result from all models for a given scenario"""
        df = pd.DataFrame()
        
        for key, value in self.matilda_scenarios[scenario].items():
            s = value[dict_name][result_name]
            s.name = key
            df = pd.concat([df, s], axis=1)

        df.index = pd.to_datetime(df.index)
        df.index.name = 'TIMESTAMP'
        print(f'{result_name} extracted for {scenario}')

        return df

    def prepare_data_for_plot(self):
        """Prepare all necessary data for plotting"""
        # Extract results from MATILDA scenarios
        self.runoff = {'SSP2': self.get_matilda_result_all_models('SSP2', 'total_runoff'),
                      'SSP5': self.get_matilda_result_all_models('SSP5', 'total_runoff')}

        self.evaporation = {'SSP2': self.get_matilda_result_all_models('SSP2', 'actual_evaporation'),
                           'SSP5': self.get_matilda_result_all_models('SSP5', 'actual_evaporation')}

        self.precipitation = {'SSP2': self.get_matilda_result_all_models('SSP2', 'total_precipitation'),
                             'SSP5': self.get_matilda_result_all_models('SSP5', 'total_precipitation')}

        self.glacier_area = {'SSP2': self.get_matilda_result_all_models('SSP2', 'glacier_area', dict_name='glacier_rescaling'),
                            'SSP5': self.get_matilda_result_all_models('SSP5', 'glacier_area', dict_name='glacier_rescaling')}

        self.snow_melt = {'SSP2': self.get_matilda_result_all_models('SSP2', 'snow_melt_on_glaciers'),
                         'SSP5': self.get_matilda_result_all_models('SSP5', 'snow_melt_on_glaciers')}

        self.ice_melt = {'SSP2': self.get_matilda_result_all_models('SSP2', 'ice_melt_on_glaciers'),
                        'SSP5': self.get_matilda_result_all_models('SSP5', 'ice_melt_on_glaciers')}

        self.off_melt = {'SSP2': self.get_matilda_result_all_models('SSP2', 'melt_off_glaciers'),
                        'SSP5': self.get_matilda_result_all_models('SSP5', 'melt_off_glaciers')}

        # Melt off glacier is always snow melt
        self.snow_melt['SSP2'] = self.snow_melt['SSP2'] + self.off_melt['SSP2']
        self.snow_melt['SSP5'] = self.snow_melt['SSP5'] + self.off_melt['SSP5']

        # Glacier area starts one year earlier
        self.adjust_startdate(self.glacier_area)

        # Convert Temp from K to °C
        for scenario in ['SSP2', 'SSP5']:
            for col in self.tas[scenario]:
                self.tas[scenario][col] = self.tas[scenario][col] - 273.15

        self.df_era5['temp'] = self.df_era5['temp'] - 273.15

        # Process melt data
        self.process_melt_data()

    def process_melt_data(self):
        """Process and aggregate melt data for plotting"""
        self.melt_ssp2 = pd.DataFrame()
        self.melt_ssp2['ssp2_avg_snow'] = self.snow_melt['SSP2'].mean(axis=1)
        self.melt_ssp2['ssp2_avg_ice'] = self.ice_melt['SSP2'].mean(axis=1)
        self.melt_ssp2 = self.melt_ssp2.resample('YE').sum()
        self.melt_ssp2.index = self.melt_ssp2.index.map(lambda x: x.replace(month=1, day=1))

        self.melt_ssp5 = pd.DataFrame()
        self.melt_ssp5['ssp5_avg_snow'] = self.snow_melt['SSP5'].mean(axis=1)
        self.melt_ssp5['ssp5_avg_ice'] = self.ice_melt['SSP5'].mean(axis=1)
        self.melt_ssp5['ssp5_avg_off'] = self.off_melt['SSP5'].mean(axis=1)
        self.melt_ssp5 = self.melt_ssp5.resample('YE').sum()
        self.melt_ssp5.index = self.melt_ssp5.index.map(lambda x: x.replace(month=1, day=1))

        self.melt_diff = pd.DataFrame()
        self.melt_diff['diff_snow'] = self.melt_ssp5['ssp5_avg_snow'] - self.melt_ssp2['ssp2_avg_snow']
        self.melt_diff['diff_ice'] = self.melt_ssp5['ssp5_avg_ice'] - self.melt_ssp2['ssp2_avg_ice']

    def custom_formatter_abs_val(self, value, _):
        """Format axis labels for absolute values"""
        return f'{abs(value):.0f}'

    def annotate_final_val(self, ax, y_axes, y_text, text, unit):
        """Annotate the final value on a plot"""
        ax.annotate(f'{text:.2f} {unit}',
                    xy=(1, y_axes), xytext=(55, y_text), va='center', ha='right',
                    fontsize=8,
                    xycoords=('axes fraction', 'data'), textcoords=('offset points', 'data'),
                    arrowprops=self.arrow_props)

    def annote_final_val_lines(self, ax, unit, min_dist=0):
        """Annotate final values for all lines on an axis"""
        # First collect all y-values
        yvals = []
        for line in ax.lines:
            ls = line.get_ls()
            if ls == ':' or ls == '--':
                ydata = line.get_ydata()
                yvals.append(ydata[-1])

        # Sort them and ensure minimum distance if supplied (lower values have to move)
        yvals.sort(reverse=True)
        for i, y in enumerate(yvals):
            if i > 0:
                diff = yvals[i-1] - yvals[i]
                if diff < min_dist:
                    yvals[i] = yvals[i] + diff - min_dist
            self.annotate_final_val(ax, y, yvals[i], y, unit)

    def df2long(self, df, val_name, intv_sum=None, intv_mean='YE', rolling=None, cutoff=None):
        """Resamples dataframes and converts them into long format for seaborn plots"""
        if intv_sum is not None:
            df = df.resample(intv_sum, label='right').sum()

        df = df.resample(intv_mean, label='right').mean()

        if rolling is not None:
            df = df.rolling(rolling).mean()

        if cutoff is not None:
            df = df.loc[cutoff:]

        # Adjust the index to assign the first day of the period
        if intv_mean == 'YE':
            df.index = df.index.map(lambda x: x.replace(month=1, day=1))
        elif intv_mean == 'M':
            df.index = df.index.map(lambda x: x.replace(day=1))

        # Reset index to turn the timestamp index into a column
        df = df.reset_index()

        # Melt the dataframe into long format
        df = df.melt('TIMESTAMP', var_name='model', value_name=val_name)

        return df

    def add_cmip_ensemble(self, param_scenarios, val_name, ylabel, ax, ylim=None, target=None,
                          target_color='black', linestyle='solid', rolling=None,
                          cutoff=None, intv_sum='YE', ylabel_pad=0):
        """Add CMIP ensemble to a plot"""
        colors = ['orange', 'dodgerblue']
        col_dict = {key: value for key, value in zip(param_scenarios.keys(), colors)}

        linestyles = ['dotted', 'dashed']
        ls_dict = {key: value for key, value in zip(param_scenarios.keys(), linestyles)}

        for i in param_scenarios.keys():
            df_pred = self.df2long(param_scenarios[i], val_name, intv_sum=intv_sum,
                                  intv_mean='YE', rolling=rolling, cutoff=cutoff)
            sns.lineplot(data=df_pred, x='TIMESTAMP', y=val_name,
                         color=target_color, ax=ax, linestyle=ls_dict[i])

        ax.set_ylabel(ylabel, labelpad=ylabel_pad)

        if ylim is not None:
            ax.set_ylim(ylim)

        if target is not None:
            ax.plot(target, linewidth=1.5, c=target_color)

    def ensemble_max(self, param_scenarios, val_name, rolling=None, cutoff=None, intv_sum='YE'):
        """Calculate ensemble maximum with confidence interval"""
        all_dfs = []

        # Step 1: Concatenate all scenarios into one long dataframe
        for i in param_scenarios.keys():
            df_pred = self.df2long(param_scenarios[i], val_name, intv_sum=intv_sum,
                                  intv_mean='YE', rolling=rolling, cutoff=cutoff)
            df_pred["scenario"] = i
            all_dfs.append(df_pred)

        df_all = pd.concat(all_dfs)

        # Step 2: Compute ensemble mean and 95% CI at each timestamp
        grouped = df_all.groupby("TIMESTAMP")[val_name]
        df_ci = grouped.agg(['mean', 'count', 'std']).reset_index()
        df_ci['ci95'] = stats.t.ppf(0.975, df_ci['count'] - 1) * (df_ci['std'] / np.sqrt(df_ci['count']))

        # Step 3: Add mean + upper confidence bound
        df_ci['upper'] = df_ci['mean'] + df_ci['ci95']

        # Step 4: Return the maximum of the upper bound
        return round(df_ci['upper'].max())

    def plot_summary(self, rolling=None, save_path=None, show=False):
        """Create the main summary plot with all components"""
        print("Creating MATILDA summary plot...")
        
        # Set up the figure with gridspec
        gridspec = dict(hspace=0.0, height_ratios=[1, 2, 4, 1])
        figure, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw=gridspec)

        # ----- Glacierized Area (top panel) -----
        print("Plotting glacierized area...")
        ax0l = axs[0]
        self.add_cmip_ensemble(param_scenarios=self.glacier_area, val_name='glac_area', 
                       ylabel='Glacierized\nArea (km²)',
                       ax=ax0l, target_color='darkviolet', ylabel_pad=10)

        # Annotate final glacier values with percentage
        for line in ax0l.lines:
            ls = line.get_ls()
            if ls == ':' or ls == '--':
                ydata = line.get_ydata()
                last_val = ydata[-1]
                perc_val = last_val / max(ydata) * 100
                label = r"{:.0f} km² ({:.0f}\%)".format(last_val, perc_val)
                ax0l.annotate(label,
                            xy=(1, last_val), xytext=(55, min(max(ydata) * 0.9, last_val * 4)),
                            va='center', ha='right', fontsize=8,
                            xycoords=('axes fraction', 'data'), textcoords=('offset points', 'data'),
                            arrowprops=self.arrow_props)

        # 50% line
        half_y = ax0l.get_ylim()[1] / 2
        ax0l.axhline(y=half_y, color='lightgrey', linestyle=':', linewidth=1)
        ax0l.text(dt.datetime(2001, 1, 1), half_y, f"50%", ha='left', va='bottom', size=8, color='grey')

        # ----- Snow & Ice Melt (second panel) -----
        print("Plotting snow & ice melt...")
        ax1l = axs[1]
        
        # Create stacked plots for both scenarios
        col = ["#eaeaea", "#d1e3ff"]  # Colors for snow and ice melt
        ax1l.stackplot(self.melt_ssp5.index, self.melt_ssp5['ssp5_avg_snow'], 
                   self.melt_ssp5['ssp5_avg_ice'], colors=col)
        ax1l.stackplot(self.melt_ssp2.index, self.melt_ssp2['ssp2_avg_snow']*-1, 
                   self.melt_ssp2['ssp2_avg_ice']*-1, colors=col)

        ax1l.axhline(y=0, color='white', linestyle='-')  # Zero line

        # Plot differences
        ax1l.plot(self.melt_diff.index, self.melt_diff['diff_snow'], color='#b6b6b6')
        ax1l.plot(self.melt_diff.index, self.melt_diff['diff_ice'], color='#a1bceb')

        # Auto-scale y-axis to match data range
        ymax_ax1l = max(max(self.melt_ssp5['ssp5_avg_snow'] + self.melt_ssp5['ssp5_avg_ice']),
                max(self.melt_ssp2['ssp2_avg_snow'] + self.melt_ssp2['ssp2_avg_ice']))
        ymax_ax1l_upper = round(ymax_ax1l*1.55, 1)  # Add space for legend
        ymax_ax1l_lower = round(-ymax_ax1l*1.1, 1)
        
        ax1l.set_ylim(ymax_ax1l_lower, ymax_ax1l_upper)
        ax1l.set_ylabel('Melt (mm/a)', labelpad=10)

        # Annotate final values
        y = self.melt_ssp5['ssp5_avg_snow'].iloc[-1]
        self.annotate_final_val(ax1l, y, ymax_ax1l*0.3, abs(y), 'mm')
        y = self.melt_ssp5['ssp5_avg_ice'].iloc[-1]+self.melt_ssp5['ssp5_avg_snow'].iloc[-1]
        self.annotate_final_val(ax1l, y, ymax_ax1l*0.8, abs(y), 'mm')
        y = self.melt_ssp2['ssp2_avg_snow'].iloc[-1]*-1
        self.annotate_final_val(ax1l, y, ymax_ax1l*-0.3, abs(y), 'mm')
        y = (self.melt_ssp2['ssp2_avg_ice'].iloc[-1]+self.melt_ssp2['ssp2_avg_snow'].iloc[-1])*-1
        self.annotate_final_val(ax1l, y, ymax_ax1l*-0.8, abs(y), 'mm')

        ax1l.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):.0f}')

        # ----- Runoff & Precipitation (third panel) -----
        print("Plotting runoff & precipitation...")
        ax2l = axs[2]

        # Auto-scale y-axis to match data range
        ymax_ax2l = max(self.ensemble_max(param_scenarios=self.runoff, val_name='runoff', 
                           rolling=rolling, cutoff='2000-12-31'),
                self.ensemble_max(param_scenarios=self.evaporation, val_name='eva', 
                          rolling=rolling, cutoff='2000-12-31'),
                self.ensemble_max(param_scenarios=self.precipitation, val_name='prec', 
                          rolling=rolling, cutoff='2000-12-31'))
        ymax_ax2l = ymax_ax2l * 1.1  # Some space for the legend

        # Plot runoff with observations
        if rolling is not None:
            obs_rs = self.obs['Qobs'].resample('YE').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
        else:
            obs_rs = self.obs['Qobs'].resample('YE').agg(pd.Series.sum, skipna=False).mean()

        self.add_cmip_ensemble(param_scenarios=self.runoff, val_name='runoff', ylabel=' (mm/a)',
                       ylim=(0, ymax_ax2l), target=obs_rs, target_color='blue',
                       ax=ax2l, rolling=rolling, cutoff='2000-12-31')

        # Plot evaporation
        self.add_cmip_ensemble(param_scenarios=self.evaporation, val_name='eva', ylabel=' (mm/a)',
                       target=None, target_color='green',
                       ax=ax2l, rolling=rolling, cutoff='2000-12-31')

        # Plot precipitation
        self.add_cmip_ensemble(param_scenarios=self.precipitation, val_name='prec', ylabel=' (mm/a)',
                       target=None, target_color='darkgrey',
                       ax=ax2l, rolling=rolling, cutoff='2000-12-31', ylabel_pad=5)

        self.annote_final_val_lines(ax2l, 'mm', min_dist=100)

        # Add observation data rectangles
        for index, row in self.obs.dropna().iterrows():
            start = mdates.date2num(row['Date'])
            ax2l.add_patch(Rectangle((start, 0), width=1, height=ymax_ax2l, alpha=0.1, 
                         label='_obs_data', zorder=0))

        ax2l.axvline(dt.datetime(2022, 12, 31), color='salmon')  # Present day line

        # ----- Temperature (bottom panel) -----
        print("Plotting temperature...")
        ax3l = axs[3]

        # Process ERA5 temperature for plotting
        if rolling is not None:
            era5_temp_rs = self.df_era5['temp'].resample('YE').agg(pd.Series.mean, skipna=False).rolling(rolling, min_periods=2).mean()
        else:
            era5_temp_rs = self.df_era5['temp'].resample('YE').agg(pd.Series.mean, skipna=False).mean()

        self.add_cmip_ensemble(param_scenarios=self.tas, val_name='temp', ylabel='Temp. (°C)',
                       target=era5_temp_rs, target_color='red',
                       ax=ax3l, rolling=rolling, cutoff='2000-12-31', intv_sum=None, ylabel_pad=10)
        
        self.annote_final_val_lines(ax3l, '°C')

        # Add 0°C line
        ax3l.axhline(y=0, color='lightgrey', linestyle=':', linewidth=1)
        ax3l.text(dt.datetime(2001, 1, 1), 0, f"0 °C", ha='left', va='bottom', size=8, color='grey')

        ax3l.axvline(dt.datetime(2022, 12, 31), color='salmon')  # Present day line

        # ----- Create legends -----
        print("Creating legends...")

        # Snow & Ice melt legend
        ax1l.legend(['Snow Melt', 'Ice Melt', '_Snow', '_Ice', '_White', 'SSP5-SSP2', 'SSP5-SSP2'], 
                ncol=2, fontsize="8", loc="upper left", frameon=False)

        # Runoff, evaporation, precipitation legend
        ax2l_legend = ax2l.legend(['_SSP2', 'Runoff', '_SSP5', '_CI5', '_Runoff',
                      '_SSP2', 'Evaporation', '_SSP5', '_CI5',
                      '_SSP2', 'Precipitation', '_SSP5', '_CI5',
                      'Runoff observation period'],
                      ncol=4, fontsize="8", loc="upper left", frameon=True)
        ax2l_legend.get_frame().set_facecolor('white')
        ax2l_legend.get_frame().set_edgecolor('white')

        # Scenario legend
        scenario_legend = ax3l.legend(['SSP2 Scenario', '_ci1', 'SSP5 Scenario', '_ci2'],
                        loc="lower right", bbox_to_anchor=(1, -0.8), ncol=2,
                        frameon=True)

        # ----- Add text annotations -----
        style = dict(size=8, color='black')
        ax1l.text(dt.datetime(2001, 1, 1), 5, f"SSP5", ha='left', va='bottom', **style)
        ax1l.text(dt.datetime(2001, 1, 1), -6, f"SSP2", ha='left', va='top', **style)
        ax1l.text(dt.datetime(2099, 1, 1), 120, f"Diff +", ha='right', va='bottom', **style)
        ax1l.text(dt.datetime(2099, 1, 1), -120, f"Diff -", ha='right', va='top', **style)

        if rolling is not None:
            ax2l.text(dt.datetime(2100, 1, 1), 0, f"{rolling} year rolling mean", 
                  ha='right', va='bottom', style='italic', **style)
            ax3l.text(dt.datetime(2100, 1, 1), era5_temp_rs.dropna().min(), 
                  f"{rolling} year rolling mean", ha='right', va='bottom', style='italic', **style)

        # ----- Final formatting -----
        print("Final formatting...")

        ax3l.xaxis.set_major_locator(mdates.YearLocator(base=10))

        for ax in axs:
            ax.margins(x=0)
            ax.set_xlim([dt.datetime(2000, 1, 1), None])

        for ax in [ax1l, ax2l]:
            ax.grid(axis='y', color='lightgrey', linestyle='--', dashes=(5, 5))

        ax3l.set_xlabel("")
        plt.suptitle(f"MATILDA Summary", fontweight='bold', fontsize=16)
        figure.tight_layout()

        # Save figure if path is provided
        if save_path:
            figure.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")

        print(">> DONE! <<")
                        
        return figure

def custom_df_matilda(dic, scenario, var, resample_freq=None):
    """
    Takes a dictionary of model outputs and returns a combined dataframe of a specific variable for a given scenario.
    Parameters
    -------
    dic : dict
        A nested dictionary of model outputs.
        The outer keys are scenario names and the inner keys are model names.
        The corresponding values are dictionaries containing two keys:
        'model_output' (DataFrame): containing model outputs for a given scenario and model
        'glacier_rescaling' (DataFrame): containing glacier properties for a given scenario and model
    scenario : str
        The name of the scenario to select from the dictionary.
    var : str
        The name of the variable to extract from the model output DataFrame.
    resample_freq : str, optional
        The frequency of the resulting time series data.
        Defaults to None (i.e. no resampling).
        If provided, should be in pandas resample frequency string format.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined data of the specified variable for the selected scenario
        and models. The DataFrame is indexed by the time steps of the original models.
        The columns are the names of the models in the selected scenario.
    Raises
    -------
    ValueError
        If the provided  var  string is not one of the following: ['avg_temp_catchment', 'avg_temp_glaciers',
        'evap_off_glaciers', 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers', 'snow_off_glaciers',
        'rain_on_glaciers', 'snow_on_glaciers', 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
        'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers', 'ice_melt_on_glaciers', 'snow_melt_on_glaciers',
        'refreezing_ice', 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation', 'total_precipitation',
        'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'runoff_ratio', 'total_runoff', 'glacier_area',
        'glacier_elev', 'smb_water_year', 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum', 'surplus',
        'glacier_melt_perc', 'glacier_mass_mmwe', 'glacier_vol_m3', 'glacier_vol_perc']
    """
    out1_cols = ['avg_temp_catchment',
                 'avg_temp_glaciers',
                 'evap_off_glaciers',
                 'prec_off_glaciers',
                 'prec_on_glaciers',
                 'rain_off_glaciers',
                 'snow_off_glaciers',
                 'rain_on_glaciers',
                 'snow_on_glaciers',
                 'snowpack_off_glaciers',
                 'soil_moisture',
                 'upper_groundwater',
                 'lower_groundwater',
                 'melt_off_glaciers',
                 'melt_on_glaciers',
                 'ice_melt_on_glaciers',
                 'snow_melt_on_glaciers',
                 'refreezing_ice',
                 'refreezing_snow',
                 'total_refreezing',
                 'SMB',
                 'actual_evaporation',
                 'total_precipitation',
                 'total_melt',
                 'runoff_without_glaciers',
                 'runoff_from_glaciers',
                 'runoff_ratio',
                 'total_runoff']

    out2_cols = ['glacier_area',
                 'glacier_elev',
                 'smb_water_year',
                 'smb_scaled',
                 'smb_scaled_capped',
                 'smb_scaled_capped_cum',
                 'surplus',
                 'glacier_melt_perc',
                 'glacier_mass_mmwe',
                 'glacier_vol_m3',
                 'glacier_vol_perc']

    if var in out1_cols:
        output_df = 'model_output'
    elif var in out2_cols:
        output_df = 'glacier_rescaling'
    else:
        raise ValueError("var needs to be one of the following strings: " +
                         str([i for i in [out1_cols, out2_cols]]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        df = dic[scenario][model][output_df]
        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()
    # Resample time series
    if resample_freq is not None:
        if output_df == 'glacier_rescaling':
            if resample_freq == '10YE':
                if var in ['glacier_area', 'glacier_elev']:
                    combined_df = combined_df.resample(resample_freq).mean()
                else:
                    combined_df = combined_df.resample(resample_freq).sum()
        else:
            if var in ['avg_temp_catchment', 'avg_temp_glaciers']:
                combined_df = combined_df.resample(resample_freq).mean()
            else:
                combined_df = combined_df.resample(resample_freq).sum()

    return combined_df


from tools.helpers import matilda_vars
import plotly.graph_objects as go
from tools.helpers import confidence_interval

def plot_ci_matilda(var, dic, resample_freq='YE', show=False):
    """
    A function to plot multi-model mean and confidence intervals of a given variable for two different scenarios.
    Parameters:
    -----------
    var: str
        The variable to plot.
    dic: dict, optional (default=matilda_scenarios)
        A dictionary containing the scenarios as keys and the dataframes as values.
    resample_freq: str, optional (default='YE')
        The resampling frequency to apply to the data.
    show: bool, optional (default=False)
        Whether to show the resulting plot or not.
    Returns:
    --------
    go.Figure
        A plotly figure object containing the mean and confidence intervals for the given variable in the two selected scenarios.
    """

    if var is None:
        var = 'total_runoff'       # Default if nothing selected

    # SSP2
    df1 = custom_df_matilda(dic, scenario='SSP2', var=var, resample_freq=resample_freq)
    df1_ci = confidence_interval(df1)
    # SSP5
    df2 = custom_df_matilda(dic, scenario='SSP5', var=var, resample_freq=resample_freq)
    df2_ci = confidence_interval(df2)

    fig = go.Figure([
        # SSP2
        go.Scatter(
            name='SSP2',
            x=df1_ci.index,
            y=round(df1_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkorange'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df1_ci.index,
            y=round(df1_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df1_ci.index,
            y=round(df1_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(255, 165, 0, 0.3)',
            fill='tonexty',
            showlegend=False
        ),

        # SSP5
        go.Scatter(
            name='SSP5',
            x=df2_ci.index,
            y=round(df2_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkblue'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df2_ci.index,
            y=round(df2_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df2_ci.index,
            y=round(df2_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=matilda_vars[var][0] + ' [' + matilda_vars[var][1] + ']',
        title={'text': '<b>' + matilda_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Arial'}},
        legend={'font': {'size': 18, 'family': 'Arial'}},
        hovermode='x',
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Set the background color to white
        margin=dict(l=10, r=10, t=90, b=10),  # Adjust the margins to remove space around the plot
        xaxis=dict(gridcolor='lightgrey'),  # set the grid color of x-axis to lightgrey
        yaxis=dict(gridcolor='lightgrey'),  # set the grid color of y-axis to lightgrey
    )
    fig.update_yaxes(rangemode='tozero')

    # show figure
    if show:
        fig.show()
    else:
        return fig
    
    
def matilda_dash(app,dic,fig_count=4,
                 default_vars=['total_runoff', 'total_precipitation', 'runoff_from_glaciers', 'glacier_area'],
                 display_mode='inLine'):
    
    # If number of default variables does not match the number of figures use variables in default order
    if fig_count != len(default_vars):
        default_vars = list(matilda_vars.keys())

    
    # Create separate callback functions for each dropdown menu and graph combination
    for i in range(fig_count):
        @app.callback(
            Output(f'line-plot-{i}', 'figure'),
            Input(f'arg-dropdown-{i}', 'value'),
            Input(f'freq-dropdown-{i}', 'value')
        )
        def update_figure(selected_arg, selected_freq, i=i):
            fig = plot_ci_matilda(selected_arg,dic=dic, resample_freq=selected_freq+'E')  # adding 'E' to resample freq due to deprecated warning (e.g. 'YE' instead of 'Y')
            return fig

    # Define the dropdown menus and figures
    dropdowns_and_figures = []
    for i in range(fig_count):
        arg_dropdown = dcc.Dropdown(
            id=f'arg-dropdown-{i}',
            options=[{'label': matilda_vars[var][0], 'value': var} for var in matilda_vars.keys()],
            value=default_vars[i],
            clearable=False,
            style={'width': '400px', 'fontFamily': 'Arial', 'fontSize': 15}
        )
        freq_dropdown = dcc.Dropdown(
            id=f'freq-dropdown-{i}',
            options=[{'label': freq, 'value': freq} for freq in ['M', 'Y', '10Y']],
            value='Y',
            clearable=False,
            style={'width': '100px'}
        )
        dropdowns_and_figures.append(
            html.Div([
                html.Div([
                    html.Label("Variable:"),
                    arg_dropdown,
                ], style={'display': 'inline-block', 'margin-right': '30px'}),
                html.Div([
                    html.Label("Frequency:"),
                    freq_dropdown,
                ], style={'display': 'inline-block'}),
                dcc.Graph(id=f'line-plot-{i}'),
            ])
        )
    # Combine the dropdown menus and figures into a single layout
    app.layout = html.Div(dropdowns_and_figures)

from tools.helpers import custom_df_indicators
from tools.indicators import indicator_vars

def plot_ci_indicators(var, dic, plot_type='line', show=False):
    """
    A function to plot multi-model mean and confidence intervals of a given variable for two different scenarios.
    Parameters:
    -----------
    var: str
        The variable to plot.
    dic: dict, optional (default=matilda_scenarios)
        A dictionary containing the scenarios as keys and the dataframes as values.
    plot_type: str, optional (default='line')
        Whether the plot should be a line or a bar plot.
    show: bool, optional (default=False)
        Whether to show the resulting plot or not.
    Returns:
    --------
    go.Figure
        A plotly figure object containing the mean and confidence intervals for the given variable in the two selected scenarios.
    """

    if var is None:
        var = 'total_runoff'       # Default if nothing selected

    # SSP2
    df1 = custom_df_indicators(dic, scenario='SSP2', var=var)
    df1_ci = confidence_interval(df1)
    # SSP5
    df2 = custom_df_indicators(dic, scenario='SSP5', var=var)
    df2_ci = confidence_interval(df2)

    if plot_type == 'line':
        fig = go.Figure([
        # SSP2
        go.Scatter(
            name='SSP2',
            x=df1_ci.index,
            y=round(df1_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkorange'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df1_ci.index,
            y=round(df1_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df1_ci.index,
            y=round(df1_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(255, 165, 0, 0.3)',
            fill='tonexty',
            showlegend=False
        ),

        # SSP5
        go.Scatter(
            name='SSP5',
            x=df2_ci.index,
            y=round(df2_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkblue'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df2_ci.index,
            y=round(df2_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df2_ci.index,
            y=round(df2_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    elif plot_type == 'bar':
        fig = go.Figure([
            # SSP2
            go.Bar(
                name='SSP2',
                x=df1_ci.index,
                y=round(df1_ci['mean'], 2),
                marker=dict(color='darkorange'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=round(df1_ci['mean'] - df1_ci['ci_lower'], 2),
                    arrayminus=round(df1_ci['ci_upper'] - df1_ci['mean'], 2),
                    color='grey'
                )
            ),
            # SSP5
            go.Bar(
                name='SSP5',
                x=df2_ci.index,
                y=round(df2_ci['mean'], 2),
                marker=dict(color='darkblue'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=round(df2_ci['mean'] - df2_ci['ci_lower'], 2),
                    arrayminus=round(df2_ci['ci_upper'] - df2_ci['mean'], 2),
                    color='grey'
                )
            )
        ])
    else:
        raise ValueError("Invalid property specified for 'plot_type'. Choose either 'line' or 'bar'")

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=indicator_vars[var][0] + ' [' + indicator_vars[var][1] + ']',
        title={'text': '<b>' + indicator_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Palatino'}},
        legend={'font': {'size': 18, 'family': 'Palatino'}},
        hovermode='x',
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Set the background color to white
        margin=dict(l=10, r=10, t=90, b=10),  # Adjust the margins to remove space around the plot
        xaxis=dict(gridcolor='lightgrey'),  # set the grid color of x-axis to lightgrey
        yaxis=dict(gridcolor='lightgrey'),  # set the grid color of y-axis to lightgrey
    )
    fig.update_yaxes(rangemode='tozero')

    # show figure
    if show:
        fig.show()
    else:
        return fig
    

def matilda_indicators_dash(app, indicator_data, fig_count=4,
                             default_vars=['peak_day', 'melt_season_length', 'potential_aridity', 'spei12'],
                             default_types=['line', 'line', 'line', 'bar']):
    
    for i in range(fig_count):
        @app.callback(
            Output(f'line-plot-{i}', 'figure'),
            Input(f'arg-dropdown-{i}', 'value'),
            Input(f'type-dropdown-{i}', 'value')
        )
        def update_figure(selected_arg, selected_type, i=i):
            fig = plot_ci_indicators(selected_arg, indicator_data, selected_type)
            return fig

    dropdowns_and_figures = []
    for i in range(fig_count):
        arg_dropdown = dcc.Dropdown(
            id=f'arg-dropdown-{i}',
            options=[{'label': indicator_vars[var][0], 'value': var} for var in indicator_vars.keys()],
            value=default_vars[i],
            clearable=False,
            style={'width': '400px', 'fontFamily': 'Palatino', 'fontSize': 15}
        )
        type_dropdown = dcc.Dropdown(
            id=f'type-dropdown-{i}',
            options=[{'label': lab, 'value': val} for lab, val in [('Line', 'line'), ('Bar', 'bar')]],
            value=default_types[i],
            clearable=False,
            style={'width': '150px'}
        )
        dropdowns_and_figures.append(
            html.Div([
                html.Div([
                    html.Label("Variable:"),
                    arg_dropdown,
                ], style={'display': 'inline-block', 'margin-right': '30px'}),
                html.Div([
                    html.Label("Plot Type:"),
                    type_dropdown,
                ], style={'display': 'inline-block'}),
                dcc.Graph(id=f'line-plot-{i}'),
            ])
        )
    
    app.layout = html.Div(dropdowns_and_figures)


def ensemble_mean(matilda_scenarios, result_name, dict_name="model_output"):
    """
    Compute the ensemble mean for a specific variable across all models in the scenarios.

    Parameters:
        matilda_scenarios (dict): Original scenario dictionary with nested model outputs.
        result_name (str): Name of the target variable to extract (e.g., 'total_runoff').
        dict_name (str): Name of the dictionary in the model output to look for results (default: 'model_output').

    Returns:
        dict: A dictionary with emission scenarios as keys and the ensemble mean time series (as pandas Series) as values.
    """

    def get_matilda_result_all_models(scenario, result_name, dict_name="model_output"):
        df = pd.DataFrame()

        for key, value in matilda_scenarios[scenario].items():
            s = value[dict_name][result_name]
            s.name = key
            df = pd.concat([df, s], axis=1)

        df.index = pd.to_datetime(df.index)
        df.index.name = "TIMESTAMP"
        print(f"{result_name} extracted for {scenario}")

        return df

    # Extract the result data for each scenario and compute the ensemble mean
    mean_results = {}
    for scenario in matilda_scenarios.keys():
        # Get the DataFrame of all models for the given scenario and variable
        df = get_matilda_result_all_models(scenario, result_name, dict_name)
        # Compute the ensemble mean across models (columns)
        mean_results[scenario] = df.mean(axis=1)

    return mean_results


def plot_annual_cycles(matilda_scenarios, scenarios=("SSP2", "SSP5"), save_path=None):
    """
    Creates a 2-column by 4-row plot of annual cycles for the given variables,
    comparing two scenarios side by side.

    Parameters:
        runoff (dict): Ensemble mean dictionary for runoff.
        snow_melt (dict): Ensemble mean dictionary for snowmelt.
        ice_melt (dict): Ensemble mean dictionary for ice melt.
        aet (dict): Ensemble mean dictionary for actual evaporation.
        scenarios (tuple): Tuple of two scenarios to compare (default: ("SSP2", "SSP5"))

    Returns:
        None: Displays the subplot figure.
    """

    runoff = ensemble_mean(matilda_scenarios, "total_runoff")
    snow_melt = ensemble_mean(matilda_scenarios, "snow_melt_on_glaciers")
    ice_melt = ensemble_mean(matilda_scenarios, "ice_melt_on_glaciers")
    off_melt = ensemble_mean(matilda_scenarios, "melt_off_glaciers")
    aet = ensemble_mean(matilda_scenarios, "actual_evaporation")

    # Melt off glacier is always snow melt:
    snow_melt["SSP2"] = snow_melt["SSP2"] + off_melt["SSP2"]
    snow_melt["SSP5"] = snow_melt["SSP5"] + off_melt["SSP5"]
    
    variables = {
        "Total Runoff": ("YlGnBu", runoff),
        "Snowmelt": ("Blues", snow_melt),
        "Ice Melt": ("Blues", ice_melt),
        "AET": ("Oranges", aet),
    }

    short_month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(4, 2, figsize=(8, 12), sharex=True)

    for row_idx, (var_name, (cmap, var_dict)) in enumerate(variables.items()):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            data = var_dict[scenario]

            # Prepare the data
            df = data.reset_index()
            df.columns = ["date", var_name]
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            # Group and pivot
            monthly_values = df.groupby(["year", "month"])[var_name].sum().reset_index()
            grid = monthly_values.pivot(index="month", columns="year", values=var_name)

            # Plot
            c = ax.pcolormesh(grid.index, grid.columns, grid.T, shading="nearest", cmap=cmap)

            # Colorbar
            cbar = fig.colorbar(c, ax=ax, ticks=[
                round(grid.min().min() / 10) * 10,
                round(grid.max().max() / 10) * 10
            ])
            c.set_clim(round(grid.min().min() / 10) * 10, round(grid.max().max() / 10) * 10)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("mm", labelpad=-13, rotation=90, fontsize=12)

            # Titles and labels
            if col_idx == 0:
                ax.set_ylabel(var_name, fontsize=16)
            if row_idx == 0:
                ax.set_title(scenario, fontsize=18)

            # Customizing ticks
            ax.set_xticks([1, 7, 12])
            ax.set_xticklabels(["Jan", "Jul", "Dec"], fontsize=13)
            ax.set_yticks([grid.columns[0], grid.columns[len(grid.columns) // 2], grid.columns[-1]])
            ax.set_yticklabels(["2000", "2050", "2100"], fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()

    
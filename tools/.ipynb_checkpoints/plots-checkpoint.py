import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import pandas as pd
import warnings
from matplotlib.legend import Legend
import probscale
from tools.helpers import read_era5l



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
plt.rcParams['axes.labelsize']: 15
#plt.rcParams['axes.linewidth'] = 1.0

# get style for matplotlib plots
# plt_style = ast.literal_eval(config['CONFIG']['PLOT_STYLE'])


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
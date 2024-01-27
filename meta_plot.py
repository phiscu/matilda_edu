import pandas as pd
import configparser
from tools.helpers import pickle_to_dict, parquet_to_dict, read_yaml

import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates

###
"""
load definitions
"""


def get_matilda_result_all_models(scenario, result_name, dict='model_output'):
    df = pd.DataFrame()

    for key, value in matilda_scenarios[scenario].items():
        s = value[dict][result_name]
        s.name = key
        df = pd.concat([df, s], axis=1)

    df.index = pd.to_datetime(df.index)
    df.index.name = 'TIMESTAMP'
    print(f'{result_name} extracted for {scenario}')

    return df


def custom_formatter_abs_val(value, _):
    return f'{abs(value):.0f}'


def annotate_final_val(ax, value, unit):
    ax.annotate(f'{abs(value):.2f} {unit}',
                xy=(1, value), xytext=(60, 0), va='center', ha='right',
                fontsize=8,
                xycoords=('axes fraction', 'data'), textcoords='offset points',
                arrowprops=arrow_props)


def annote_final_val_lines(ax, unit):
    for line in ax.lines:
        ls = line.get_ls()
        if ls == ':' or ls == '--':
            ydata = line.get_ydata()
            annotate_final_val(ax, ydata[-1], unit)

###
"""
get config/settings
"""
warnings.filterwarnings(action='ignore')

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']

settings = read_yaml(dir_output + 'settings.yml')

###
"""
Load data that has been determined in other notebooks
"""
obs = pd.read_csv(dir_input + 'obs_runoff_example.csv')
obs.index = pd.to_datetime(obs['Date'])
obs["Qobs"] = obs["Qobs"] * 86400 / (settings['area_cat'] * 1000000) * 1000

df_era5 = pd.read_csv(dir_output + 'ERA5L.csv', **{
        'usecols':      ['temp', 'prec', 'dt'],
        'index_col':    'dt',
        'parse_dates':  ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})

tas = parquet_to_dict(f"{dir_output}cmip6/adjusted/tas_parquet")
pr = parquet_to_dict(f"{dir_output}cmip6/adjusted/pr_parquet")

matilda_scenarios = parquet_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")

###
"""
Prepare data for plot
"""

runoff = {'SSP2': get_matilda_result_all_models('SSP2', 'total_runoff'),
          'SSP5': get_matilda_result_all_models('SSP5', 'total_runoff')}

glacier_area = {'SSP2': get_matilda_result_all_models('SSP2', 'glacier_area', dict='glacier_rescaling'),
                'SSP5': get_matilda_result_all_models('SSP5', 'glacier_area', dict='glacier_rescaling')}

snow_melt = {'SSP2': get_matilda_result_all_models('SSP2', 'snow_melt_on_glaciers'),
             'SSP5': get_matilda_result_all_models('SSP5', 'snow_melt_on_glaciers')}

ice_melt = {'SSP2': get_matilda_result_all_models('SSP2', 'ice_melt_on_glaciers'),
            'SSP5': get_matilda_result_all_models('SSP5', 'ice_melt_on_glaciers')}

###

melt_ssp2 = pd.DataFrame()

melt_ssp2['ssp2_avg_snow'] = snow_melt['SSP2'].mean(axis=1)
melt_ssp2['ssp2_avg_ice'] = ice_melt['SSP2'].mean(axis=1)
melt_ssp2 = melt_ssp2.resample('Y').sum()

melt_ssp5 = pd.DataFrame()
melt_ssp5['ssp5_avg_snow'] = snow_melt['SSP5'].mean(axis=1)
melt_ssp5['ssp5_avg_ice'] = ice_melt['SSP5'].mean(axis=1)
melt_ssp5 = melt_ssp5.resample('Y').sum()

###


def df2long(df, val_name, intv_sum=None, intv_mean='Y', rolling=None, cutoff=None):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""
    if intv_sum is not None:
        df = df.resample(intv_sum).sum()

    df = df.resample(intv_mean).mean()

    if rolling is not None:
        df = df.rolling(rolling).mean()

    if cutoff is not None:
        df = df.loc[cutoff:]

    df = df.reset_index()
    df = df.melt('TIMESTAMP', var_name='model', value_name=val_name)

    print(df)

    return df


def add_cmip_ensemble(param_scenarios, val_name, ylabel, ax, ylim=None, target=None, target_color='black', linestyle='solid', rolling=None, cutoff=None, intv_sum='Y'):
    # Define color palette
    colors = ['orange', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(param_scenarios.keys(), colors)}

    # Define color palette
    linestyles = ['dotted', 'dashed']
    # create a new dictionary with the same keys but new values from the list
    ls_dict = {key: value for key, value in zip(param_scenarios.keys(), linestyles)}


    for i in param_scenarios.keys():
        df_pred = df2long(param_scenarios[i], val_name, intv_sum=intv_sum, intv_mean='Y', rolling=rolling, cutoff=cutoff)
        # sns.lineplot(data=df_pred, x='TIMESTAMP', y=val_name, color=col_dict[i], ax=ax, linestyle=linestyle)
        sns.lineplot(data=df_pred, x='TIMESTAMP', y=val_name, color=target_color, ax=ax, linestyle=ls_dict[i])
    ax.set(xlabel='Year', ylabel=ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    if target is not None:
        target_plot = ax.plot(target, linewidth=1.5, c=target_color)


###

plt.rcParams["font.family"] = "Arial"

rolling = 5

arrow_props = dict(facecolor='grey', edgecolor='grey', arrowstyle='-', linewidth=0.5)



# --- START PLOT ---
gridspec = dict(hspace=0.0, height_ratios=[1, 3, 3, 1])
figure, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)


# -> fill box: glacerized area
ax0l = axs[0]
add_cmip_ensemble(param_scenarios=glacier_area, val_name='glac_area', ylabel='Glacerized\nArea (%)', ylim=(0,50),
                  ax=ax0l, target_color='darkviolet') #linestyle='dotted')

annote_final_val_lines(ax0l, '%')
print("Shrink the glacier")

# -> fill box: snow & ice melt
ax1l = axs[1]

col = ["#eaeaea", "#d1e3ff"]
ax1l.stackplot(melt_ssp5.index, melt_ssp5['ssp5_avg_snow'], melt_ssp5['ssp5_avg_ice'], colors = col)
ax1l.stackplot(melt_ssp5.index, melt_ssp2['ssp2_avg_snow'] * -1, melt_ssp2['ssp2_avg_ice'] * -1, colors = col)
ax1l.axhline(y=0, color='white', linestyle='-')
ax1l.set_ylim(-145, 145)
ax1l.set_ylabel('Melt (mm/a)')

annotate_final_val(ax1l, melt_ssp5['ssp5_avg_snow'][-1], 'mm')
annotate_final_val(ax1l, melt_ssp5['ssp5_avg_ice'][-1]+melt_ssp5['ssp5_avg_snow'][-1], 'mm')
annotate_final_val(ax1l, melt_ssp2['ssp2_avg_snow'][-1]*-1, 'mm')
annotate_final_val(ax1l, melt_ssp2['ssp2_avg_ice'][-1]*-1+melt_ssp2['ssp2_avg_snow'][-1]*-1, 'mm')

ax1l.yaxis.set_major_formatter(custom_formatter_abs_val)

print("Melt snow & ice")

# -> fill box: runoff & prec
ax2l = axs[2]

obs_rs = obs['Qobs'].resample('Y').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
add_cmip_ensemble(param_scenarios=runoff, val_name='runoff', ylabel=' (mm/a)', ylim=(0,1750),
                  target=obs_rs, target_color='blue',
                  ax=ax2l, rolling=rolling, cutoff='2020-12-31')

era5_prec_rs = df_era5['prec'].resample('Y').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
add_cmip_ensemble(param_scenarios=pr, val_name='prec', ylabel='Runoff / Precipitation (mm/a)', ylim=(0,1750),
                  target=era5_prec_rs, target_color='grey',  #linestyle='dashed',
                  ax=ax2l, rolling=rolling, cutoff='2022-12-31')

annote_final_val_lines(ax2l, 'mm')

print("Let the water run ")

# -> fill box: Temperature
ax3l = axs[3]

era5_temp_rs = df_era5['temp'].resample('Y').agg(pd.Series.mean, skipna=False).rolling(rolling, min_periods=2).mean()
add_cmip_ensemble(param_scenarios=tas, val_name='temp', ylabel='Temp. (K)',
                  target=era5_temp_rs, target_color='red',  #linestyle='dashdot',
                  ax=ax3l, rolling=rolling, cutoff='2022-12-31', intv_sum=None)

annote_final_val_lines(ax3l, 'K')
print("Turn on some heat")

# -> create legend
ax1l.legend(['Snow Melt', 'Ice Melt'], ncol=1, fontsize="8",
            loc="lower right",  # bbox_to_anchor=(0.43, 1.3),
            frameon=False)

# ax1l.legend(['SSP2 Scenario', '_ci1', 'SSP5 Scenario', '_ci2','Runoff',
#              '_SSP2 Scenario', '_ci1', '_SSP5 Scenario', '_ci2','Precipitation'],
#             loc="upper center", bbox_to_anchor=(0.38, -0.15), ncol=4,
#             frameon=False)  # First legend --> Workaround as seaborn lists CIs in legend
#
scenario_legend = ax3l.legend(['SSP2 Scenario', '_ci1', 'SSP5 Scenario', '_ci2'],
                            loc="lower right", bbox_to_anchor=(1, -0.8), ncol=2,
                            frameon=True)  # First legend --> Workaround as seaborn lists CIs in legend
for handle in scenario_legend.legendHandles:
    handle.set_color('black')


print("Legend ready")

# -> Add texts to the plot
style = dict(size=8, color='black')
ax1l.text(dt.datetime(1982, 1, 1), 5, f"SSP5", ha='left', va='bottom', **style)
ax1l.text(dt.datetime(1982, 1, 1), -5, f"SSP2", ha='left', va='top', **style)
ax2l.text(dt.datetime(1982, 1, 1), 1600, f"{rolling} year rolling mean", **style)

# -> final polish: modify x-Axis and show current date line
ax3l.xaxis.set_major_locator(mdates.YearLocator(base=10))

for ax in axs:
    ax.axvline(dt.datetime(2024, 1, 27), color='salmon')
    ax.margins(x=0)

for ax in [ax1l,ax2l]:
    ax.grid(axis='y', color='lightgrey', linestyle='--', dashes=(5, 5))


figure.tight_layout(rect=[0, 0.02, 1, 1])  # Make some room at the bottom

# --- SHOW ---
plt.show()







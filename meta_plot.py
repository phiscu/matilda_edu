import pandas as pd
import configparser
from tools.helpers import parquet_to_dict, read_yaml

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


def annotate_final_val(ax, y, value, unit):
    ax.annotate(f'{abs(value):.2f} {unit}',
                xy=(1, value), xytext=(55, y), va='center', ha='right',
                fontsize=8,
                xycoords=('axes fraction', 'data'), textcoords=('offset points','data'),
                arrowprops=arrow_props)


def annote_final_val_lines(ax, unit, min_dist=0):
    # first collect all y-values
    yvals = []
    for line in ax.lines:
        ls = line.get_ls()
        if ls == ':' or ls == '--':
            ydata = line.get_ydata()
            yvals.append(ydata[-1])

    # sort them and ensure minimum distance if supplied (lower values have to move)
    yvals.sort(reverse=True)
    for i,y in enumerate(yvals):
        if i > 0:
            diff = yvals[i-1] - yvals[i]
            if diff < min_dist:
                yvals[i] = yvals[i] + diff - min_dist
        annotate_final_val(ax, yvals[i], y, unit)


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
# pr = parquet_to_dict(f"{dir_output}cmip6/adjusted/pr_parquet")

matilda_scenarios = parquet_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")

###
"""
Prepare data for plot
"""

runoff = {'SSP2': get_matilda_result_all_models('SSP2', 'total_runoff'),
          'SSP5': get_matilda_result_all_models('SSP5', 'total_runoff')}

evaporation = {'SSP2': get_matilda_result_all_models('SSP2', 'actual_evaporation'),
               'SSP5': get_matilda_result_all_models('SSP5', 'actual_evaporation')}

precipitation = {'SSP2': get_matilda_result_all_models('SSP2', 'total_precipitation'),
                 'SSP5': get_matilda_result_all_models('SSP5', 'total_precipitation')}

glacier_area = {'SSP2': get_matilda_result_all_models('SSP2', 'glacier_area', dict='glacier_rescaling'),
                'SSP5': get_matilda_result_all_models('SSP5', 'glacier_area', dict='glacier_rescaling')}

snow_melt = {'SSP2': get_matilda_result_all_models('SSP2', 'snow_melt_on_glaciers'),
             'SSP5': get_matilda_result_all_models('SSP5', 'snow_melt_on_glaciers')}

ice_melt = {'SSP2': get_matilda_result_all_models('SSP2', 'ice_melt_on_glaciers'),
            'SSP5': get_matilda_result_all_models('SSP5', 'ice_melt_on_glaciers')}

###

# glacier_area_proc = pd.DataFrame()
# glacier_area_proc['ssp2'] = glacier_area['SSP2'].mean(axis=1)
# glacier_area_proc['ssp5'] = glacier_area['SSP5'].mean(axis=1)
# glacier_area_proc['ssp2_adj'] = glacier_area_proc['ssp2'] / max(glacier_area_proc['ssp2']) * 100
# glacier_area_proc['ssp5_adj'] = glacier_area_proc['ssp5'] / max(glacier_area_proc['ssp5']) * 100

for scenario in ['SSP2','SSP5']:
    for col in glacier_area[scenario]:
        glacier_area[scenario][col] = glacier_area[scenario][col] / max(glacier_area[scenario][col]) * 100



melt_ssp2 = pd.DataFrame()
melt_ssp2['ssp2_avg_snow'] = snow_melt['SSP2'].mean(axis=1)
melt_ssp2['ssp2_avg_ice'] = ice_melt['SSP2'].mean(axis=1)
melt_ssp2 = melt_ssp2.resample('Y').sum()

melt_ssp5 = pd.DataFrame()
melt_ssp5['ssp5_avg_snow'] = snow_melt['SSP5'].mean(axis=1)
melt_ssp5['ssp5_avg_ice'] = ice_melt['SSP5'].mean(axis=1)
melt_ssp5 = melt_ssp5.resample('Y').sum()

melt_diff = pd.DataFrame()
melt_diff['diff_snow'] = melt_ssp5['ssp5_avg_snow'] - melt_ssp2['ssp2_avg_snow']
melt_diff['diff_ice'] = melt_ssp5['ssp5_avg_ice'] - melt_ssp2['ssp2_avg_ice']

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

#--- START PLOT ---
gridspec = dict(hspace=0.0, height_ratios=[0.5, 2, 3, 0.5])
figure, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True, gridspec_kw=gridspec)

# -> fill box: glacerized area
ax0l = axs[0]
add_cmip_ensemble(param_scenarios=glacier_area, val_name='glac_area', ylabel='Glacerized\nArea (%)', ylim=(0,105),
                  ax=ax0l, target_color='darkviolet')

# ax0l.plot(glacier_area_proc.index, glacier_area_proc['ssp2_adj'], color='darkviolet')
# ax0l.plot(glacier_area_proc.index, glacier_area_proc['ssp5_adj'], color='darkviolet')

annote_final_val_lines(ax0l, '%')

print("Shrink the glacier")

# -> fill box: snow & ice melt
ax1l = axs[1]
# ax1bl = axs[2]

col = ["#eaeaea", "#d1e3ff"]
ax1l.stackplot(melt_ssp5.index, melt_ssp5['ssp5_avg_snow'], melt_ssp5['ssp5_avg_ice'], colors = col)
ax1l.stackplot(melt_ssp2.index, melt_ssp2['ssp2_avg_snow']*-1, melt_ssp2['ssp2_avg_ice']*-1, colors = col)

ax1l.axhline(y=0, color='white', linestyle='-')

ax1l.plot(melt_diff.index, melt_diff['diff_snow'], color='#b6b6b6')
ax1l.plot(melt_diff.index, melt_diff['diff_ice'], color='#a1bceb') # darkblue

ax1l.set_ylim(-149, 149)
ax1l.set_ylabel('Melt (mm/a)')


y = melt_ssp5['ssp5_avg_snow'][-1]
annotate_final_val(ax1l, y, y, 'mm')
y = melt_ssp5['ssp5_avg_ice'][-1]+melt_ssp5['ssp5_avg_snow'][-1]
annotate_final_val(ax1l, y, y, 'mm')
y = melt_ssp2['ssp2_avg_snow'][-1]*-1
annotate_final_val(ax1l, y, y, 'mm')
y = (melt_ssp2['ssp2_avg_ice'][-1]+melt_ssp2['ssp2_avg_snow'][-1])*-1
annotate_final_val(ax1l, y, y, 'mm')

ax1l.yaxis.set_major_formatter(custom_formatter_abs_val)

print("Melt snow & ice")

# -> fill box: runoff & prec
ax2l = axs[2]

obs_rs = obs['Qobs'].resample('Y').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
add_cmip_ensemble(param_scenarios=runoff, val_name='runoff', ylabel=' (mm/a)', ylim=(0,1550),
                  target=obs_rs, target_color='blue',
                  ax=ax2l, rolling=rolling, cutoff='2020-12-31')

add_cmip_ensemble(param_scenarios=evaporation, val_name='eva', ylabel=' (mm/a)', #ylim=(0,1750),
                  target=None, target_color='green',  #linestyle='dashed',
                  ax=ax2l, rolling=rolling, cutoff='1981-12-31')

add_cmip_ensemble(param_scenarios=precipitation, val_name='prec', ylabel=' (mm/a)', #ylim=(0,1750),
                  target=None, target_color='darkgrey',  #linestyle='dashed',
                  ax=ax2l, rolling=rolling, cutoff='1981-12-31')

# era5_prec_rs = df_era5['prec'].resample('Y').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
# add_cmip_ensemble(param_scenarios=pr, val_name='prec', ylabel='Runoff / Precipitation (mm/a)', ylim=(0,1750),
#                   target=era5_prec_rs, target_color='grey',  #linestyle='dashed',
#                   ax=ax2l, rolling=rolling, cutoff='2022-12-31')

annote_final_val_lines(ax2l, 'mm', min_dist=100)

print("Let the water run ")

# -> fill box: Temperature
ax3l = axs[3]

era5_temp_rs = df_era5['temp'].resample('Y').agg(pd.Series.mean, skipna=False).rolling(rolling, min_periods=2).mean()
add_cmip_ensemble(param_scenarios=tas, val_name='temp', ylabel='Temp. (K)',
                  target=era5_temp_rs, target_color='red',
                  ax=ax3l, rolling=rolling, cutoff='2022-12-31', intv_sum=None)

annote_final_val_lines(ax3l, 'K')
print("Turn on some heat")

# -> create legend
ax1l.legend(['Snow Melt', 'Ice Melt','_Snow','_Ice','_White','SSP5-SSP2','SSP5-SSP2'], ncol=2, fontsize="8",
            loc="upper left",
            frameon=False)

ax2l.legend(['_SSP2','Runoff','_SSP5','_CI5','_Runoff',
             '_SSP2','Evaporation','_SSP5','_CI5',
             '_SSP2','Precipitation','_SSP5','_CI5'],
            ncol=3, fontsize="8",
            loc="upper left",
            frameon=False)

scenario_legend = ax3l.legend(['SSP2 Scenario', '_ci1', 'SSP5 Scenario', '_ci2'],
                            loc="lower right", bbox_to_anchor=(1, -0.8), ncol=2,
                            frameon=True)  # First legend --> Workaround as seaborn lists CIs in legend
for handle in scenario_legend.legendHandles:
    handle.set_color('black')


print("Legend ready")

# -> Add texts to the plot
style = dict(size=8, color='black')
ax1l.text(dt.datetime(1982, 1, 1), 5, f"SSP5", ha='left', va='bottom', **style)
ax1l.text(dt.datetime(1982, 1, 1), -6, f"SSP2", ha='left', va='top', **style)
# ax2l.text(dt.datetime(1982, 1, 1), 80, f"{rolling} year rolling mean", **style)

# -> final polish: modify x-Axis and show current date line
ax3l.xaxis.set_major_locator(mdates.YearLocator(base=10))

for ax in axs:
    ax.axvline(dt.datetime(2024, 1, 27), color='salmon')
    ax.margins(x=0)

for ax in [ax1l,ax1l,ax2l]:
    ax.grid(axis='y', color='lightgrey', linestyle='--', dashes=(5, 5))

plt.suptitle(f"{rolling} year rolling mean")
figure.tight_layout(rect=[0, 0.02, 1, 1])  # Make some room at the bottom

# --- SHOW ---
plt.show()


###

# import plotly.io as pio
# import chart_studio
# from plotly import tools as tls
#
# chart_studio.tools.set_credentials_file(username='elgarnelo', api_key='QN8ti8vfUpnSvQfSjTAG')
#
###


# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
#
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)
#
# fig, ax = plt.subplots()
# ax.plot(t, s)
#
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()
#
# # fig.savefig("test.png")
# plt.show()
#
#
#
# # ###
# # plotly_fig = tls.mpl_to_plotly(figure)
# # plotly_fig.write_html('plotly.html', auto_open=True)
#
# np.random.seed(0)
# x, y = np.random.random((2,30))
# ###
# fig, ax = plt.subplots()
# plt.plot(x, y, 'bo')
# texts = [plt.text(x[i], y[i], 'Text%s' %i) for i in range(len(x))]
# adjust_text(texts,only_move='y', arrowprops=dict(arrowstyle='->', color='red'))
# plt.show()
import pandas as pd
import configparser
from scipy import stats
import numpy as np
from tools.helpers import parquet_to_dict, read_yaml, pickle_to_dict
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import scienceplots
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",            # or 'sans-serif', 'monospace', etc.
    "font.serif": ["Computer Modern"], # default LaTeX serif font
})


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


def annotate_final_val(ax, y_axes, y_text, text, unit):
    ax.annotate(f'{text:.2f} {unit}',
                xy=(1, y_axes), xytext=(55, y_text), va='center', ha='right',
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
        annotate_final_val(ax, y, yvals[i], y, unit)


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
obs['Date'] = obs.index
obs["Qobs"] = obs["Qobs"] * 86400 / (settings['area_cat'] * 1000000) * 1000

df_era5 = pd.read_csv(dir_output + 'ERA5L.csv', **{
        'usecols':      ['temp', 'prec', 'dt'],
        'index_col':    'dt',
        'parse_dates':  ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})

# Adjust start date:
df_era5.index = pd.to_datetime(df_era5.index)  # Convert the index to datetime
df_era5 = df_era5[df_era5.index >= '2000-01-01']  # Filter rows where the date is >= 2000-01-01
obs = obs[obs.index >= '2000-01-01']  # Filter rows where the date is >= 2000-01-01

tas = pickle_to_dict(f"{dir_output}cmip6/adjusted/tas.pickle")
pr = pickle_to_dict(f"{dir_output}cmip6/adjusted/pr.pickle")

# Adjust start date
def adjust_startdate(data_dict, start_date='2000-01-01'):
    for key in data_dict:
        df = data_dict[key]
        df.index = pd.to_datetime(df.index)  # Convert the index to datetime
        data_dict[key] = df[df.index >= start_date]  # Filter rows where the date is >= start_date


# Apply the function to both dictionaries
adjust_startdate(tas)
adjust_startdate(pr)

matilda_scenarios = pickle_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")

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

off_melt = {'SSP2': get_matilda_result_all_models('SSP2', 'melt_off_glaciers'),
            'SSP5': get_matilda_result_all_models('SSP5', 'melt_off_glaciers')}

# Melt off glacier is always snow melt:
snow_melt['SSP2'] = snow_melt['SSP2'] + off_melt['SSP2']
snow_melt['SSP5'] = snow_melt['SSP5'] + off_melt['SSP5']

# Glacier area starts one year earlier
adjust_startdate(glacier_area)

###

for scenario in ['SSP2','SSP5']:
    # # convert glacier area from km² in % of max value (-> starting from ~100% and declining)
    # for col in glacier_area[scenario]:
    #     glacier_area[scenario][col] = glacier_area[scenario][col] / max(glacier_area[scenario][col]) * 100
    # convert Temp. from K to °C
    for col in tas[scenario]:
        tas[scenario][col] = tas[scenario][col] - 273.15

df_era5['temp'] = df_era5['temp'] - 273.15

melt_ssp2 = pd.DataFrame()
melt_ssp2['ssp2_avg_snow'] = snow_melt['SSP2'].mean(axis=1)
melt_ssp2['ssp2_avg_ice'] = ice_melt['SSP2'].mean(axis=1)
# melt_ssp2['ssp2_avg_off'] = off_melt['SSP2'].mean(axis=1)
melt_ssp2 = melt_ssp2.resample('Y').sum()

melt_ssp2.index = melt_ssp2.index.map(lambda x: x.replace(month=1, day=1))  # Assign the first day of the year

melt_ssp5 = pd.DataFrame()
melt_ssp5['ssp5_avg_snow'] = snow_melt['SSP5'].mean(axis=1)
melt_ssp5['ssp5_avg_ice'] = ice_melt['SSP5'].mean(axis=1)
melt_ssp5['ssp5_avg_off'] = off_melt['SSP5'].mean(axis=1)
melt_ssp5 = melt_ssp5.resample('Y').sum()

melt_ssp5.index = melt_ssp5.index.map(lambda x: x.replace(month=1, day=1))  # Assign the first day of the year

melt_diff = pd.DataFrame()
melt_diff['diff_snow'] = melt_ssp5['ssp5_avg_snow'] - melt_ssp2['ssp2_avg_snow']
melt_diff['diff_ice'] = melt_ssp5['ssp5_avg_ice'] - melt_ssp2['ssp2_avg_ice']
# melt_diff['diff_off'] = melt_ssp5['ssp5_avg_off'] - melt_ssp2['ssp2_avg_off']

###

def df2long(df, val_name, intv_sum=None, intv_mean='Y', rolling=None, cutoff=None):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""
    if intv_sum is not None:
        df = df.resample(intv_sum, label='right').sum()  # Align resampling to the start of the period

    df = df.resample(intv_mean, label='right').mean()  # Align resampling to the start of the period

    if rolling is not None:
        df = df.rolling(rolling).mean()

    if cutoff is not None:
        df = df.loc[cutoff:]

    # Adjust the index to assign the first day of the period
    if intv_mean == 'Y':
        df.index = df.index.map(lambda x: x.replace(month=1, day=1))  # Assign the first day of the year
    elif intv_mean == 'M':
        df.index = df.index.map(lambda x: x.replace(day=1))  # Assign the first day of the month

    # Reset index to turn the timestamp index into a column
    df = df.reset_index()

    # Melt the dataframe into long format
    df = df.melt('TIMESTAMP', var_name='model', value_name=val_name)

    return df


def add_cmip_ensemble(param_scenarios, val_name, ylabel, ax, ylim=None, target=None,
                      target_color='black', linestyle='solid', rolling=None,
                      cutoff=None, intv_sum='Y', ylabel_pad=0):

    # Define color and line style palettes
    colors = ['orange', 'dodgerblue']
    col_dict = {key: value for key, value in zip(param_scenarios.keys(), colors)}

    linestyles = ['dotted', 'dashed']
    ls_dict = {key: value for key, value in zip(param_scenarios.keys(), linestyles)}

    for i in param_scenarios.keys():
        df_pred = df2long(param_scenarios[i], val_name, intv_sum=intv_sum,
                          intv_mean='Y', rolling=rolling, cutoff=cutoff)
        sns.lineplot(data=df_pred, x='TIMESTAMP', y=val_name,
                     color=target_color, ax=ax, linestyle=ls_dict[i])

    ax.set('')
    ax.set_ylabel(ylabel, labelpad=ylabel_pad)

    if ylim is not None:
        ax.set_ylim(ylim)

    if target is not None:
        ax.plot(target, linewidth=1.5, c=target_color)



def ensemble_max(param_scenarios, val_name, rolling=None, cutoff=None, intv_sum='Y'):
    all_dfs = []

    # Step 1: Concatenate all scenarios into one long dataframe
    for i in param_scenarios.keys():
        df_pred = df2long(param_scenarios[i], val_name, intv_sum=intv_sum,
                          intv_mean='Y', rolling=rolling, cutoff=cutoff)
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



### Construct main figure

# plt.rcParams["font.family"] = "Arial"

rolling = None

arrow_props = dict(facecolor='grey', edgecolor='grey', arrowstyle='-', linewidth=0.5)

#--- START PLOT ---
gridspec = dict(hspace=0.0, height_ratios=[1, 2, 4, 1])
figure, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw=gridspec)

# -> fill box: glacerized area
print("Shrink the glacier")

ax0l = axs[0]
add_cmip_ensemble(param_scenarios=glacier_area, val_name='glac_area', ylabel='Glacierized\nArea (km²)',
                  ax=ax0l, target_color='darkviolet', ylabel_pad=10)

# annote_final_val_lines(ax0l, 'km')

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
                      arrowprops=arrow_props)

# 50% line
half_y = ax0l.get_ylim()[1] / 2
ax0l.axhline(y=half_y, color='lightgrey', linestyle=':', linewidth = 1)
ax0l.text(dt.datetime(2001, 1, 1), half_y, f"50%", ha='left', va='bottom', size=8, color='grey')

# -> fill box: snow & ice melt
print("Melt snow & ice")

ax1l = axs[1]
ax1bl = axs[2]

col = ["#eaeaea", "#d1e3ff"]
ax1l.stackplot(melt_ssp5.index, melt_ssp5['ssp5_avg_snow'], melt_ssp5['ssp5_avg_ice'], colors = col)
ax1l.stackplot(melt_ssp2.index, melt_ssp2['ssp2_avg_snow']*-1, melt_ssp2['ssp2_avg_ice']*-1, colors = col)

ax1l.axhline(y=0, color='white', linestyle='-')

ax1l.plot(melt_diff.index, melt_diff['diff_snow'], color='#b6b6b6')
ax1l.plot(melt_diff.index, melt_diff['diff_ice'], color='#a1bceb') # darkblue

# Auto-scale y-axis of the stack plot to match the data range
ymax_ax1l = max(max(melt_ssp5['ssp5_avg_snow'] + melt_ssp5['ssp5_avg_ice']),
           max(melt_ssp2['ssp2_avg_snow'] + melt_ssp2['ssp2_avg_ice']))

ymax_ax1l_upper = round(ymax_ax1l*1.55, 1)     # add some space for the legend
ymax_ax1l_lower = round(-ymax_ax1l*1.1, 1)

ax1l.set_ylim(ymax_ax1l_lower, ymax_ax1l_upper)
ax1l.set_ylabel('Melt (mm/a)', labelpad=10)

y = melt_ssp5['ssp5_avg_snow'][-1]
annotate_final_val(ax1l, y, ymax_ax1l*0.3, abs(y), 'mm')
y = melt_ssp5['ssp5_avg_ice'][-1]+melt_ssp5['ssp5_avg_snow'][-1]
annotate_final_val(ax1l, y, ymax_ax1l*0.8, abs(y), 'mm')
y = melt_ssp2['ssp2_avg_snow'][-1]*-1
annotate_final_val(ax1l, y, ymax_ax1l*-0.3, abs(y), 'mm')
y = (melt_ssp2['ssp2_avg_ice'][-1]+melt_ssp2['ssp2_avg_snow'][-1])*-1
annotate_final_val(ax1l, y, ymax_ax1l*-0.8, abs(y), 'mm')

ax1l.yaxis.set_major_formatter(custom_formatter_abs_val)

# -> fill box: runoff & prec
print("Let the water cycle")

ax2l = axs[2]

# Auto-scale y-axis of the line plot to match the data range
ymax_ax2l = max(ensemble_max(param_scenarios=runoff, val_name='runoff', rolling=rolling, cutoff='2000-12-31'),
                ensemble_max(param_scenarios=evaporation, val_name='eva', rolling=rolling, cutoff='2000-12-31'),
                ensemble_max(param_scenarios=precipitation, val_name='prec', rolling=rolling, cutoff='2000-12-31'))
ymax_ax2l = ymax_ax2l * 1.1     # some space for the legend


print("- runoff")
if rolling is not None:
    obs_rs = obs['Qobs'].resample('Y').agg(pd.Series.sum, skipna=False).rolling(rolling, min_periods=2).mean()
else:
    obs_rs = obs['Qobs'].resample('Y').agg(pd.Series.sum, skipna=False).mean()

add_cmip_ensemble(param_scenarios=runoff, val_name='runoff', ylabel=' (mm/a)', ylim=(0, ymax_ax2l),
                  target=obs_rs, target_color='blue',
                  ax=ax2l, rolling=rolling, cutoff='2000-12-31')

print("- evaporation")
add_cmip_ensemble(param_scenarios=evaporation, val_name='eva', ylabel=' (mm/a)',
                  target=None, target_color='green',
                  ax=ax2l, rolling=rolling, cutoff='2000-12-31')

print("- precipitation")
add_cmip_ensemble(param_scenarios=precipitation, val_name='prec', ylabel=' (mm/a)',
                  target=None, target_color='darkgrey',
                  ax=ax2l, rolling=rolling, cutoff='2000-12-31', ylabel_pad=5)

annote_final_val_lines(ax2l, 'mm', min_dist=100)


print("Observation data")
# Iterate over Observation data without NaN values
for index, row in obs.dropna().iterrows():
    start = mdates.date2num(row['Date'])
    ax2l.add_patch(Rectangle((start, 0), width=1, height=ymax_ax2l, alpha=0.1, label='_obs_data', zorder=0))

ax2l.axvline(dt.datetime(2022, 12, 31), color='salmon')

# -> fill box: Temperature
print("Turn on some heat")

ax3l = axs[3]

if rolling is not None:
    era5_temp_rs = df_era5['temp'].resample('Y').agg(pd.Series.mean, skipna=False).rolling(rolling, min_periods=2).mean()
else:
    era5_temp_rs = df_era5['temp'].resample('Y').agg(pd.Series.mean, skipna=False).mean()

add_cmip_ensemble(param_scenarios=tas, val_name='temp', ylabel='Temp. (°C)',
                  target=era5_temp_rs, target_color='red',
                  ax=ax3l, rolling=rolling, cutoff='2000-12-31', intv_sum=None, ylabel_pad=10)
annote_final_val_lines(ax3l, '°C')

ax3l.axhline(y=0, color='lightgrey', linestyle=':', linewidth = 1)
ax3l.text(dt.datetime(2001, 1, 1), 0, f"0 °C", ha='left', va='bottom', size=8, color='grey')

ax3l.axvline(dt.datetime(2022, 12, 31), color='salmon')

# -> create legend
ax1l.legend(['Snow Melt', 'Ice Melt','_Snow','_Ice','_White','SSP5-SSP2','SSP5-SSP2'], ncol=2, fontsize="8",
            loc="upper left",
            frameon=False)

ax2l_legend = ax2l.legend(['_SSP2','Runoff','_SSP5','_CI5','_Runoff',
             '_SSP2','Evaporation','_SSP5','_CI5',
             '_SSP2','Precipitation','_SSP5','_CI5',
             'Runoff observation period',],
             ncol=4, fontsize="8",
             loc="upper left",
             frameon=True)
ax2l_legend.get_frame().set_facecolor('white')
ax2l_legend.get_frame().set_edgecolor('white')

scenario_legend = ax3l.legend(['SSP2 Scenario', '_ci1', 'SSP5 Scenario', '_ci2'],
                            loc="lower right", bbox_to_anchor=(1, -0.8), ncol=2,
                            frameon=True)  # First legend --> Workaround as seaborn lists CIs in legend

print("Legend ready")

# -> Add texts to the plot
style = dict(size=8, color='black')
ax1l.text(dt.datetime(2001, 1, 1), 5, f"SSP5", ha='left', va='bottom', **style)
ax1l.text(dt.datetime(2001, 1, 1), -6, f"SSP2", ha='left', va='top', **style)

ax1l.text(dt.datetime(2099, 1, 1), 120, f"Diff +", ha='right', va='bottom', **style)
ax1l.text(dt.datetime(2099, 1, 1), -120, f"Diff -", ha='right', va='top', **style)

if rolling is not None:
    ax2l.text(dt.datetime(2100, 1, 1), 0, f"{rolling} year rolling mean", ha='right', va='bottom', style='italic', **style)
    ax3l.text(dt.datetime(2100, 1, 1), era5_temp_rs.dropna().min(), f"{rolling} year rolling mean", ha='right', va='bottom', style='italic', **style)
# -> final polish: modify x-Axis and show current date line
ax3l.xaxis.set_major_locator(mdates.YearLocator(base=10))

for ax in axs:
    ax.margins(x=0)
    ax.set_xlim([dt.datetime(2000, 1, 1), None])  # Set lower limit to 2000, upper limit adjusts automatically

for ax in [ax1l,ax2l]:
    ax.grid(axis='y', color='lightgrey', linestyle='--', dashes=(5, 5))

ax3l.set_xlabel("")
plt.suptitle(f"MATILDA Summary", fontweight='bold', fontsize=16)
# figure.tight_layout(rect=[0, 0, 0.95, 1])  # Make space on the right (reduce right boundary from 1 to 0.95)
figure.tight_layout()

# figure.savefig("/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/HESS/figures/matilda_summary.png", dpi=300)

# --- SHOW ---
plt.show()





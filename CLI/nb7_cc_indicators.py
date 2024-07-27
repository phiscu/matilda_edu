##
from tools.helpers import pickle_to_dict, parquet_to_dict
import configparser
import os
from tools.indicators import cc_indicators
from tqdm import tqdm
import pandas as pd
from tools.helpers import dict_to_pickle, dict_to_parquet
from tools.indicators import indicator_vars
import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.io as pio
import matplotlib.font_manager as fm
path_to_palatinottf = '/home/phillip/Downloads/Palatino.ttf'
fm.fontManager.addfont(path_to_palatinottf)

# read output directory from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']

print("Importing MATILDA scenarios...")

matilda_scenarios = pickle_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")


def calculate_indicators(dic, **kwargs):
    """
    Calculate climate change indicators for all scenarios and models.
    Parameters
    ----------
    dic : dict
        Dictionary containing MATILDA outputs for all scenarios and models.
    **kwargs : optional
        Optional keyword arguments to be passed to the cc_indicators() function.
    Returns
    -------
    dict
        Dictionary with the same structure as the input but containing climate change indicators in annual resolution.
    """
    # Create an empty dictionary to store the outputs
    out_dict = {}
    # Loop over the scenarios with progress bar
    for scenario in dic.keys():
        model_dict = {}  # Create an empty dictionary to store the model outputs
        # Loop over the models with progress bar
        for model in tqdm(dic[scenario].keys(), desc=scenario):
            # Get the dataframe for the current scenario and model
            df = dic[scenario][model]['model_output']
            # Run the indicator function
            indicators = cc_indicators(df, **kwargs)
            # Store indicator time series in the model dictionary
            model_dict[model] = indicators
        # Store the model dictionary in the scenario dictionary
        out_dict[scenario] = model_dict

    return out_dict

indicator_path = f"{dir_output}cmip6/adjusted/matilda_indicators_pickle"
if os.path.exists(indicator_path):
    print("Reading Climate Change Indicators...")
    matilda_indicators = pickle_to_dict(f"{dir_output}cmip6/adjusted/matilda_indicators_pickle")
    print('Done!')
else:
    print("Calculating Climate Change Indicators...")
    matilda_indicators = calculate_indicators(matilda_scenarios)
    print("Writing Indicators To File...")
    dict_to_pickle(matilda_indicators, f"{dir_output}cmip6/adjusted/matilda_indicators_pickle")
    print('Done!')


def custom_df_indicators(dic, scenario, var):
    """
    Takes a dictionary of climate change indicators and returns a combined dataframe of a specific variable for
    a given scenario.
    Parameters
    ----------
    dic : dict
        Dictionary containing the outputs of calculate_indicators() for different scenarios and models.
    scenario : str
        Name of the selected scenario.
    var : str
        Name of the variable to extract from the DataFrame.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the selected variable from different models within the specified scenario.
    Raises
    ------
    ValueError
        If the provided variable is not one of the function outputs.
    """

    out_cols = ['max_prec_month', 'min_prec_month',
                'peak_day',
                'melt_season_start', 'melt_season_end', 'melt_season_length',
                'actual_aridity', 'potential_aridity',
                'dry_spell_days',
                'qlf_freq', 'qlf_dur', 'qhf_freq', 'qhf_dur',
                'clim_water_balance', 'spi1', 'spei1', 'spi3', 'spei3',
                'spi6', 'spei6', 'spi12', 'spei12', 'spi24', 'spei24']

    if var not in out_cols:
        raise ValueError("var needs to be one of the following strings: " +
                         str([i for i in out_cols]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        df = dic[scenario][model]
        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()

    return combined_df


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
        var = 'total_runoff'  # Default if nothing selected

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
        title={'text': '<b>' + indicator_vars[var][0] + '</b>',
               'font': {'size': 28, 'color': 'darkblue', 'family': 'Palatino'}},
        legend={'font': {'size': 18}},
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


# plot_ci_indicators(var='potential_aridity', dic=matilda_indicators, plot_type='line', show=True)


##
from scipy.stats import linregress

def calculate_linear_trend(df, column_name):
    """
    Calculate the linear trend for a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to calculate the trend for.

    Returns:
    slope (float): The slope of the linear trend line.
    intercept (float): The intercept of the linear trend line.
    p_value (float): The p-value indicating the significance of the trend.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Extract the values from the column
    y = df[column_name].values
    x = np.arange(len(y))  # Use the index as the x values

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return slope, intercept, p_value

length = confidence_interval(custom_df_indicators(matilda_indicators, 'SSP5', 'dry_spell_days'))
start = confidence_interval(custom_df_indicators(matilda_indicators, 'SSP5', 'melt_season_start'))

length['2090-01-01':'2100-12-31']['mean'].mean()

calculate_linear_trend(length.dropna(), 'mean')

##

pio.renderers.default = "browser"
app = dash.Dash()

# Create default variables for every figure
default_vars = ['peak_day', 'melt_season_length', 'potential_aridity', 'spei12']
default_types = ['line', 'line', 'line', 'bar']


# Create separate callback functions for each dropdown menu and graph combination
for i in range(4):
    @app.callback(
        Output(f'line-plot-{i}', 'figure'),
        Input(f'arg-dropdown-{i}', 'value'),
        Input(f'type-dropdown-{i}', 'value')
    )
    def update_figure(selected_arg, selected_type, i=i):
        fig = plot_ci_indicators(selected_arg, matilda_indicators, selected_type)
        return fig

# Define the dropdown menus and figures
dropdowns_and_figures = []
for i in range(4):
    arg_dropdown = dcc.Dropdown(
        id=f'arg-dropdown-{i}',
        options=[{'label': indicator_vars[var][0], 'value': var} for var in indicator_vars.keys()],
        value=default_vars[i],
        clearable=False,
        style={'width': '400px',  'fontFamily': 'Palatino', 'fontSize': 15}
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
# Combine the dropdown menus and figures into a single layout
app.layout = html.Div(dropdowns_and_figures)
# Run the app
app.run_server(debug=True, use_reloader=False, port=8051)  # Turn off reloader inside Jupyter

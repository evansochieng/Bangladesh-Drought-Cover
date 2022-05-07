#import modules
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_auth
import plotly.express as px
from dash import dash_table
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
from dash_extensions import EventListener
import base64

#Read in the file that contains bad years
sitetable = pd.read_csv("sitetable.csv")

#Define range of years
years = list(map(str, range(1981, 2019)))

#dekad dates image
dekad_dates = 'dekaddates.png'
encoded_dekad_dates = base64.b64encode(open(dekad_dates, 'rb').read())

#Add authentication my app
VALID_USERNAME_PASSWORD_PAIRS = {
    'bangladesh_project': 'drought_cover'
}


# Create the dash application
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

colors = {
    'background': '#ebebeb',
    'text': '#7FDBFF',
    'subheadingsText': "#003153",
    'outputText': "blue",
    'submitText': '#00d1d1'
}

###Define a function that will output the general DataFrames required in the project
def bangladesh_basic_analysis(location, frequency, early_window, late_window, early_weight, late_weight):
    # load data for the specific location
    rain_in = pd.read_csv(location + "_CHIRPS.csv")

    early_window = [t for t in early_window] #create list of specific specific early window range values
    late_window = [t for t in late_window] #create list of specific specific late window range values

    x_early = early_window[0]
    y_early = early_window[1]
    x_late = late_window[0]
    y_late = late_window[1]

    # create a dataframe of rainfall total for the early window
    early_rain = pd.DataFrame(rain_in.loc[x_early:y_early, :].sum(axis=0, skipna=True)).transpose()

    # create a dataframe of rainfall total for the late window
    late_rain = pd.DataFrame(rain_in.loc[x_late:y_late, :].sum(axis=0, skipna=True)).transpose()

    # Triggers and exits
    freq = float(frequency)

    early_trig = round(np.quantile(early_rain, freq))
    early_exit = round(early_rain.min(axis=1).to_list()[0])

    late_trig = round(np.quantile(late_rain, freq))
    late_exit = round(late_rain.min(axis=1).to_list()[0])

    # BMD data

    # load data
    bmd_rain_in = pd.read_csv(location + "_BMD.csv")

    # bmd rainfall total
    bmd_early_rain = pd.DataFrame(bmd_rain_in.loc[x_early:y_early, :].sum(axis=0, skipna=True)).transpose()
    bmd_late_rain = pd.DataFrame(bmd_rain_in.loc[x_late:y_late, :].sum(axis=0, skipna=True)).transpose()

    # triggers and exit
    freq = float(frequency)

    bmd_early_trig = round(np.quantile(bmd_early_rain, freq))
    bmd_early_exit = round(bmd_early_rain.min(axis=1).to_list()[0])

    bmd_late_trig = round(np.quantile(bmd_late_rain, freq))
    bmd_late_exit = round(bmd_late_rain.min(axis=1).to_list()[0])

    # PAY
    # PAY1

    # early chirps pay
    early_pay = pd.DataFrame(columns=early_rain.columns)
    for i in early_rain.columns:
        early_pay.loc[:, i] = (early_rain.loc[:, i] - early_trig) / (early_exit - early_trig)
    early_pay[early_pay < 0] = 0  # Assign 0 to the years that payout
    early_pay[early_pay > 1] = 1  # Assign 1 to the years that do not pay.

    # late chirps pay
    late_pay = pd.DataFrame(columns=late_rain.columns)
    for i in late_rain.columns:
        late_pay.loc[:, i] = (late_rain.loc[:, i] - late_trig) / (late_exit - late_trig)
    late_pay[late_pay < 0] = 0  # Assign 0 to the years that payout
    late_pay[late_pay > 1] = 1  # Assign 1 to the years that do not pay.

    # early bmd pay
    bmd_early_pay = pd.DataFrame(columns=bmd_early_rain.columns)
    for i in bmd_early_rain.columns:
        bmd_early_pay.loc[:, i] = (bmd_early_rain.loc[:, i] - bmd_early_trig) / (bmd_early_exit - bmd_early_trig)
    bmd_early_pay[bmd_early_pay < 0] = 0  # Assign 0 to the years that payout
    bmd_early_pay[bmd_early_pay > 1] = 1  # Assign 1 to the years that do not pay.

    # late bmd pay
    bmd_late_pay = pd.DataFrame(columns=bmd_late_rain.columns)
    for i in bmd_late_rain.columns:
        bmd_late_pay.loc[:, i] = (bmd_late_rain.loc[:, i] - bmd_late_trig) / (bmd_late_exit - bmd_late_trig)
    bmd_late_pay[bmd_late_pay < 0] = 0  # Assign 0 to the years that payout
    bmd_late_pay[bmd_late_pay > 1] = 1  # Assign 1 to the years that do not pay.

    # PAYOUT
    payout = pd.concat(
        [early_pay.transpose(), late_pay.transpose(), bmd_early_pay.transpose(), bmd_late_pay.transpose()], axis=1)
    payout.reset_index(inplace=True)
    columns = ['year', "CHIRPS_Early", 'CHIRPS_Late', 'BMD_Early', 'BMD_Late']
    payout.columns = columns

    # stack the columns into a single column of data using the melt()
    #This will be used in the plot
    final_payout_df = pd.melt(payout, id_vars="year")

    # Trig_Exit Table
    trig_exit_dict = {
        'Dataset/Timing': ["CHIRPS_Early", 'CHIRPS_Late', 'BMD_Early', 'BMD_Late'],
        'Trigger': [early_trig, late_trig, bmd_early_trig, bmd_late_trig],
        'Exit': [early_exit, late_exit, bmd_early_exit, bmd_late_exit]
    }

    trig_exit = pd.DataFrame(trig_exit_dict)

    return payout, final_payout_df, trig_exit


# Layout
app.layout = html.Div([
    html.Div([
        html.H1('Parameter Optimization - Validation',
                style={'textAlign': 'center', 'font-size': 40, 'font-weight': 'bold', 'color': '#503D36'})
    ]),
    html.Br(),

    html.Div(children=[
        html.Div(children=[
            html.H2('Bangladesh Deficit Rainfall', style={'font-size': 24, 'font-weight': 'bold', 'color': '#503D36'}),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']}, children=["Location"]),
                html.Div(dcc.Dropdown(id='location',
                                      options=[{'label': 'Birganj', 'value': 'Birganj'},
                                               {'label': 'Sherpur', 'value': 'Sherpur'}],
                                      value='Sherpur'))
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Indicate the payout frequency"]),
                html.Div(dcc.Input(id='freq', type='number', min=0, max=1, step=0.025, value=0.2))
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Early Window Timing (Dekad)"]),
                html.Div(dcc.RangeSlider(id='Early', min=1, max=36, step=1, marks={i: i for i in range(1, 36, 2)},
                                         value=[18, 21]))
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Late Window Timing (Dekad)"]),
                html.Div(dcc.RangeSlider(id='Late', min=1, max=36, step=1, marks={i: i for i in range(1, 36, 2)},
                                         value=[21, 23]))
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Indicate the weight for the Early Window"]),
                html.Div(dcc.Input(id='early_wgt', type='number', min=0, max=1, step=0.05, value=0.5))
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Indicate the weight for the Late Window"]),
                html.Div(dcc.Input(id='late_wgt', type='number', min=0, max=1, step=0.05, value=0.5))
            ]),

            html.Br(),

            html.Div(children=[
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=[
                               "Download Trigger, Exit, Index parameters and Payouts for the choosen parameters for the selected location"]),
                html.Button('Download', id='btn',
                            style={'textAlign': 'center', 'backgroundColor': colors['submitText']}),
                Download(id='download')
            ]),

            html.Br(),

            html.Div(children=[
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']}, children=["Dekad dates"]),
                html.Img(src='data:image/png;base64,{}'.format(encoded_dekad_dates.decode()), style={'width': '400px'})
            ])
        ], style={'width': '30%'}),

        html.Div(children=[
            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Reported Bad Years for selected location"]),
                html.Div([
                    dash_table.DataTable(id='bad-years', style_cell={ 'border': '1px solid grey' }, style_data={ 'color': 'black' },
                                         style_header={ 'border': '1px solid grey', 'fontWeight':'bold' })
                ])
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Plot of the Historic Payout for the selected location"]),
                dcc.Graph(id='historic-payout', figure={})
            ]),

            html.Div(children=[
                html.Br(),
                html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                           children=["Bad Years Matching"]),
                html.Div(id='matching', style={'whiteSpace': 'pre-line'})
            ]),

            html.Div(children=[
                html.Div(children=[
                    html.Br(),
                    html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                               children=["Triggers and Exits"]),
                    html.Div([dash_table.DataTable(id='trig-exit', style_cell={ 'border': '1px solid grey' }, style_data={ 'color': 'black' },
                                                   style_header={ 'border': '1px solid grey', 'fontWeight':'bold' })])
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div(children=[
                    html.Br(),
                    html.Label(style={"font-weight": "bold", 'color': colors['subheadingsText']},
                               children=["Pure Risk Premium"]),
                    html.Div([dash_table.DataTable(id='premium', style_cell={ 'border': '1px solid grey' }, style_data={ 'color': 'black' },
                                                   style_header={ 'border': '1px solid grey', 'fontWeight':'bold' })])
                ], style={'display': 'inline-block', 'width': '50%', 'margin-left': '50px'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'margin':'auto'})
        ], style={'margin-left': '100px'})
    ], style={'display': 'flex'})
], style={'backgroundColor': colors['background'], 'margin':'auto'})


#App callbacks
#Table outputs and plot(bar graph)
@app.callback([Output(component_id='historic-payout', component_property='figure'),
               Output(component_id='bad-years', component_property='data'),
               Output(component_id='bad-years', component_property='columns'),
               Output(component_id='trig-exit', component_property='data'),
               Output(component_id='trig-exit', component_property='columns'),
               Output(component_id='premium', component_property='data'),
               Output(component_id='premium', component_property='columns')],
              [Input('location', 'value'),
               Input('freq', 'value'),
               Input('Early', 'value'),
               Input('Late', 'value'),
               Input('early_wgt', 'value'),
               Input('late_wgt', 'value')
              ])
def bangladesh_wii_analysis(location, frequency, early_window, late_window, early_wgt, late_wgt):
    location, frequency, early_window, late_window = location, frequency, early_window, late_window
    early_weight, late_weight = early_wgt, late_wgt

    payout, final_payout_df, trig_exit = bangladesh_basic_analysis(
        location, frequency, early_window, late_window, early_weight, late_weight
    )

    # OUTPUT PLOT 1
    sitetable.fillna(0, inplace=True)
    bad = sitetable.iloc[0, 1:].to_list() if location == "Birganj" else sitetable.iloc[1, 1:-1].to_list()

    test = list(range(len(years)))
    test = [0.5 + x for x in test]
    bad_flag = [int(x) - 1981 for x in bad]

    fig = px.bar(final_payout_df, x="year", y="value", color="variable", range_y=(0, 1),
                 title="<b>Historic Payouts for {}</b>".format(location),
                 barmode='group',
                 labels={"year": "<b>YEAR</b>", "value": "<b>PERCENTAGE</b>"})

    fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, yaxis_tickformat='.0%')
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))

    for i in bad_flag:
        if i != 0:
            fig.add_vline(x=i, line_width=2, line_dash="dash", line_color="green")
    for i in test:
        fig.add_vline(x=i, line_width=0.3, line_color="grey")

    # OUTPUT TABLE 1
    ##Bad years in table form
    bad_years = pd.DataFrame(sitetable.loc[sitetable["location"] == location, :])

    years_data = bad_years.to_dict(orient='records')
    years_columns = [{'name': col, 'id': col} for col in bad_years.columns]

    ##OUTPUT TABLE 2
    ##Trig_Exit in table form
    trig_exit = trig_exit

    trig_exit_data = trig_exit.to_dict(orient='records')
    trig_exit_columns = [{'name': col, 'id': col} for col in trig_exit.columns]

    ##OUTPUT TABLE 3
    # RISK PREMIUM
    pay = payout

    pay["CHIRPS_Combined"] = pay["CHIRPS_Early"] * early_wgt + pay["CHIRPS_Late"] * late_wgt
    pay["BMD_Combined"] = pay["BMD_Early"] * early_wgt + pay["BMD_Late"] * late_wgt

    pay.iloc[:, 1:] = round(pay.iloc[:, 1:] * 100)
    pay.iloc[:, 1:] = np.where((pay.iloc[:, 1:]) > 100, 100,
                               pay.iloc[:, 1:])  # ensure the payouts doesn't go past the maximum payout

    premium = pay.iloc[:, 1:].mean(axis=0)

    premium_dict = {
        'Dataset': ['CHIRPS', 'BMD'],
        'Early Window': [round(premium["CHIRPS_Early"], 2), round(premium["BMD_Early"], 2)],
        'Late Window': [round(premium["CHIRPS_Late"], 2), round(premium["BMD_Late"], 2)],
        'Combined Windows': [round(premium["CHIRPS_Combined"], 2), round(premium["BMD_Combined"], 2)]
    }
    risk_premium = pd.DataFrame(premium_dict)

    premium_data = risk_premium.to_dict(orient='records')
    premium_columns = [{'name': col, 'id': col} for col in risk_premium.columns]

    return [
        fig,
        years_data, years_columns,
        trig_exit_data, trig_exit_columns,
        premium_data, premium_columns
    ]

#MATCHING
#String output(Matching)
@app.callback([Output(component_id='matching', component_property='children')],
              [Input('location', 'value'),
               Input('freq', 'value'),
               Input('Early', 'value'),
               Input('Late', 'value'),
               Input('early_wgt', 'value'),
               Input('late_wgt', 'value')
              ])

#define function that will output scores for matching years
def bangladesh_wii_analysis(location, frequency, early_window, late_window, early_wgt, late_wgt):
    location, frequency, early_window, late_window = location, frequency, early_window, late_window
    early_weight, late_weight = early_wgt, late_wgt

    payout, final_payout_df, trig_exit = bangladesh_basic_analysis(
        location, frequency, early_window, late_window, early_weight, late_weight
    )

    # This part does the matching i.e. A bad year mentioned by the farmers(FGD) and the year actually paid out
    # according to the set trigger and exit.

    pay = payout

    chirps_early_pay = pay.iloc[:, 1].to_frame().transpose()
    chirps_late_pay = pay.iloc[:, 2].to_frame().transpose()
    chirps_both_pay = ((chirps_early_pay * early_wgt).loc["CHIRPS_Early"] + (chirps_late_pay * late_wgt).loc[
        "CHIRPS_Late"]).to_frame().transpose()

    bmd_early_pay = pay.iloc[:, 3].to_frame().transpose()
    bmd_late_pay = pay.iloc[:, 4].to_frame().transpose()
    bmd_both_pay = ((bmd_early_pay * early_wgt).loc["BMD_Early"] + (bmd_late_pay * late_wgt).loc[
        "BMD_Late"]).to_frame().transpose()

    years = list(map(str, range(1981, 2019)))
    BadYears = pd.DataFrame(0, index=[0], columns=years)

    read_bad_years = sitetable.iloc[0, 1:].to_frame().transpose() if location == "Birganj" else sitetable.iloc[1,
                                                                                                1:].to_frame().transpose()
    read_bad_years = read_bad_years.dropna(axis=1, how='any')
    read_bad_years = read_bad_years.astype(int)

    for i in range(len(read_bad_years.columns)):
        BadYears.loc[0, read_bad_years.iloc[:, i].astype(str)] = 1

    # chirps payout match
    chirps_early_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_early_match.iloc[0, i] = 1 if (
                    (BadYears.iloc[0, i] == 1) & (chirps_early_pay.loc["CHIRPS_Early", i] > 0)) else 0

    chirps_late_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_late_match.iloc[0, i] = 1 if (
                    (BadYears.iloc[0, i] == 1) & (chirps_late_pay.loc["CHIRPS_Late", i] > 0)) else 0

    chirps_both_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_both_match.iloc[0, i] = 1 if ((BadYears.iloc[0, i] == 1) & (chirps_both_pay.loc[0, i] > 0)) else 0

    # bmd payout match
    bmd_early_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_early_match.iloc[0, i] = 1 if ((BadYears.iloc[0, i] == 1) & (bmd_early_pay.loc["BMD_Early", i] > 0)) else 0

    bmd_late_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_late_match.iloc[0, i] = 1 if ((BadYears.iloc[0, i] == 1) & (bmd_late_pay.loc["BMD_Late", i] > 0)) else 0

    bmd_both_match = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_both_match.iloc[0, i] = 1 if ((BadYears.iloc[0, i] == 1) & (bmd_both_pay.loc[0, i] > 0)) else 0

    # chirps total pay
    chirps_early_match_total = sum(chirps_early_match.loc[0, :]) / sum(BadYears.loc[0, :])
    chirps_late_match_total = sum(chirps_late_match.loc[0, :]) / sum(BadYears.loc[0, :])
    chirps_both_match_total = sum(chirps_both_match.loc[0, :]) / sum(BadYears.loc[0, :])

    # bmd total pay
    bmd_early_match_total = sum(bmd_early_match.loc[0, :]) / sum(BadYears.loc[0, :])
    bmd_late_match_total = sum(bmd_late_match.loc[0, :]) / sum(BadYears.loc[0, :])
    bmd_both_match_total = sum(bmd_both_match.loc[0, :]) / sum(BadYears.loc[0, :])

    # chirps pay years
    chirps_early_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_early_pay_year.iloc[0, i] = 1 if (chirps_early_pay.loc["CHIRPS_Early", i] > 0) else 0

    chirps_late_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_late_pay_year.iloc[0, i] = 1 if (chirps_late_pay.loc["CHIRPS_Late", i] > 0) else 0

    chirps_both_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        chirps_both_pay_year.iloc[0, i] = 1 if (chirps_both_pay.loc[0, i] > 0) else 0

    # bmd pay years
    bmd_early_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_early_pay_year.iloc[0, i] = 1 if (bmd_early_pay.loc["BMD_Early", i] > 0) else 0

    bmd_late_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_late_pay_year.iloc[0, i] = 1 if (bmd_late_pay.loc["BMD_Late", i] > 0) else 0

    bmd_both_pay_year = pd.DataFrame(np.nan, index=[0], columns=years)
    for i in range(len(years)):
        bmd_both_pay_year.iloc[0, i] = 1 if (bmd_both_pay.loc[0, i] > 0) else 0

    # HSS formula is given by =2(ad-bc)/[(a+c)(c+d)+(a+b)(b+d)],below we are defining a,b,c and d in order to compute HSS Score
    # CHIRPS
    chirps_early_a = int(sum(chirps_early_match.iloc[0, :]))
    chirps_early_b = int(sum(chirps_early_pay)) - int(sum(chirps_early_match.iloc[0, :]))
    chirps_early_c = int(sum(BadYears.iloc[0, :])) - int(sum(chirps_early_match.iloc[0, :]))
    chirps_early_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(chirps_early_match.iloc[0, :]))
    chirps_early_HSS = 2 * ((chirps_early_a * chirps_early_d) - (chirps_early_b * chirps_early_c)) / (
                ((chirps_early_a + chirps_early_c) * (chirps_early_c + chirps_early_d)) + (
                    (chirps_early_a + chirps_early_b) * (chirps_early_b + chirps_early_d)))

    chirps_late_a = int(sum(chirps_late_match.iloc[0, :]))
    chirps_late_b = int(sum(chirps_late_pay)) - int(sum(chirps_late_match.iloc[0, :]))
    chirps_late_c = int(sum(BadYears.iloc[0, :])) - int(sum(chirps_late_match.iloc[0, :]))
    chirps_late_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(chirps_late_match.iloc[0, :]))
    chirps_late_HSS = 2 * (chirps_late_a * chirps_late_d - chirps_late_b * chirps_late_c) / (
                (chirps_late_a + chirps_late_c) * (chirps_late_c + chirps_late_d) + (chirps_late_a + chirps_late_b) * (
                    chirps_late_b + chirps_late_d))

    chirps_both_a = int(sum(chirps_both_match.iloc[0, :]))
    chirps_both_b = int(sum(chirps_both_pay)) - int(sum(chirps_both_match.iloc[0, :]))
    chirps_both_c = int(sum(BadYears.iloc[0, :])) - int(sum(chirps_both_match.iloc[0, :]))
    chirps_both_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(chirps_both_match.iloc[0, :]))
    chirps_both_HSS = 2 * (chirps_both_a * chirps_both_d - chirps_both_b * chirps_both_c) / (
                (chirps_both_a + chirps_both_c) * (chirps_both_c + chirps_both_d) + (chirps_both_a + chirps_both_b) * (
                    chirps_both_b + chirps_both_d))

    # BMD
    bmd_early_a = int(sum(bmd_early_match.iloc[0, :]))
    bmd_early_b = int(sum(bmd_early_pay)) - int(sum(bmd_early_match.iloc[0, :]))
    bmd_early_c = int(sum(BadYears.iloc[0, :])) - int(sum(bmd_early_match.iloc[0, :]))
    bmd_early_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(bmd_early_match.iloc[0, :]))
    bmd_early_HSS = 2 * (bmd_early_a * bmd_early_d - bmd_early_b * bmd_early_c) / (
                (bmd_early_a + bmd_early_c) * (bmd_early_c + bmd_early_d) + (bmd_early_a + bmd_early_b) * (
                    bmd_early_b + bmd_early_d))

    bmd_late_a = int(sum(bmd_late_match.iloc[0, :]))
    bmd_late_b = int(sum(bmd_late_pay)) - int(sum(bmd_late_match.iloc[0, :]))
    bmd_late_c = int(sum(BadYears.iloc[0, :])) - int(sum(bmd_late_match.iloc[0, :]))
    bmd_late_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(bmd_late_match.iloc[0, :]))
    bmd_late_HSS = 2 * (bmd_late_a * bmd_late_d - bmd_late_b * bmd_late_c) / (
                (bmd_late_a + bmd_late_c) * (bmd_late_c + bmd_late_d) + (bmd_late_a + bmd_late_b) * (
                    bmd_late_b + bmd_late_d))

    bmd_both_a = int(sum(bmd_both_match.iloc[0, :]))
    bmd_both_b = int(sum(bmd_both_pay)) - int(sum(bmd_both_match.iloc[0, :]))
    bmd_both_c = int(sum(BadYears.iloc[0, :])) - int(sum(bmd_both_match.iloc[0, :]))
    bmd_both_d = int(len(years)) - 2 * (int(sum(BadYears.iloc[0, :]))) + int(sum(bmd_both_match.iloc[0, :]))
    bmd_both_HSS = 2 * (bmd_both_a * bmd_both_d - bmd_both_b * bmd_both_c) / (
                (bmd_both_a + bmd_both_c) * (bmd_both_c + bmd_both_d) + (bmd_both_a + bmd_both_b) * (
                    bmd_both_b + bmd_both_d))

    # Print out the percentage identified and HSS scores

    return ['Percentage of ' + location + '\'s Bad Years Identified and Heidke Skill Scores (HSS):\n' +
            '\n'
            'CHIRPS Early Window ' + str(round(chirps_early_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(chirps_early_HSS, 3)) + '\n'
            'CHIRPS Late Window ' + str(round(chirps_late_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(chirps_late_HSS, 3)) + '\n'
            'CHIRPS Both Window ' + str(round(chirps_both_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(chirps_both_HSS, 3)) + '\n'
            '\n'
            'BMD Early Window ' + str(round(bmd_early_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(bmd_early_HSS, 3)) + '\n'
            'BMD Late Window ' + str(round(bmd_late_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(bmd_late_HSS, 3)) + '\n'
            'BMD Both Window ' + str(round(bmd_both_match_total * 100, 2)) + '% of hit - HSS: ' + str(round(bmd_both_HSS, 3))
            ]

@app.callback([Output("download", "data")],
              [Input("btn", "n_clicks")],
              [State('location', 'value'),
               State('freq', 'value'),
               State('Early', 'value'),
               State('Late', 'value'),
               State('early_wgt', 'value'),
               State('late_wgt', 'value')])
               #prevent_initial_call=True)
def bangladesh_download_analysis(location, frequency, early_window, late_window, early_wgt, late_wgt, n_nlicks):
    location, frequency, early_window, late_window = location, frequency, early_window, late_window
    early_weight, late_weight = early_wgt, late_wgt

    payout, final_payout_df, trig_exit = bangladesh_basic_analysis(
        location, frequency, early_window, late_window, early_weight, late_weight
    )

    # Make a directory where the files will be downloaded
    #os.chdir('C:/Users/HP/')
    #os.makedirs(os.path.join('Desktop', location, '_Bangladesh_Deficit_Files'), exist_ok=True)

    # Create the first file csv file that will be downloaded
    # (i)Historical Payout File
    pay = payout  # the second the payout df
    pay["CHIRPS_Combined"] = pay['CHIRPS_Early'] * early_wgt + pay['CHIRPS_Late'] * late_wgt
    pay["BMD_Combined"] = pay['BMD_Early'] * early_wgt + pay['BMD_Late'] * late_wgt

    pay.iloc[:, 1:] = round(pay.iloc[:, 1:] * 100)
    pay.iloc[:, 1:] = np.where((pay.iloc[:, 1:]) > 100, 100,
                               pay.iloc[:, 1:])  # ensure the payouts doesn't go past the maximum payout

    # Bad Years
    BadYears = pd.DataFrame(0, index=[0], columns=years)

    read_bad_years = sitetable.iloc[0, 1:].to_frame().transpose() if location == "Birganj" else sitetable.iloc[1, 1:-1].to_frame().transpose()
    read_bad_years = read_bad_years.dropna(axis=1, how='any')
    read_bad_years = read_bad_years.astype(int)

    for i in range(len(read_bad_years.columns)):
        BadYears.loc[0, read_bad_years.iloc[:, i].astype(str)] = 1

    # Add bad years as a column to the pay df
    pay["BadYears"] = BadYears.loc[0, :].values.transpose()
    pay = pd.DataFrame(pay)

    #os.chdir(os.path.join('C:/Users/HP/', 'Desktop', location, '_Bangladesh_Deficit_Files'))
    filename = location + '_payouts_deficit.csv'
    return [send_data_frame(pay.to_csv, filename)]


#Run the app
if __name__ == '__main__':
    app.run_server()



import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dash import no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from yahooquery import Ticker
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_daq as daq
from scipy.stats import norm
import datetime
from datetime import date
from dateutil.rrule import rrule, DAILY
import math
from math import exp, log, sqrt


def black_scholes(S, K, duration, r, a, ot, time_factor,st):
    # S = Spot Price
    # K = Strike Price
    # T  = duration /365
    # r = risk free rate
    # a = implied volatility
    # time_factor will be 1 day and will reduce the time to expiration by 1
    # ot stands for option type(calls or puts)
    T = ((duration+1) - (time_factor * st)) / 365
    d1 = (log(S / K) + ((r + ((a ** 2) / 2)) * T)) / (a * (sqrt(abs(T))))
    d2 = d1 - (a * (sqrt(abs(T))))

    if ot == 'calls':
        result = (S * (norm.cdf(d1))) - (K * exp((-r) * T) * (norm.cdf(d2)))
    if ot == 'puts':
        result = (K * exp((-r) * T) * (norm.cdf(-d2))) - (S * (norm.cdf(-d1)))

    return result


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# APP MAIN LAYOUT
# Margin Around Containers 10px
# Margin Around between buttons 5px
app.layout = html.Div(id='page-container',

                      children=[

                          html.Div(id='content-container',
                                   children=[
                                       dbc.Row(
                                           dbc.Col(children=[
                                               html.H1(children='Option Strategy Dashboard'),
                                           ],
                                           ),
                                       ),

                                       dbc.Row(
                                           [
                                               # Left Panel Begins
                                               dbc.Col(
                                                   [
                                                       dbc.Row(style={'padding': '10px'},
                                                               children=
                                                               [
                                                                   dbc.Col([
                                                                       dbc.Card(
                                                                           [
                                                                               dbc.CardBody(
                                                                                   [
                                                                                       html.H5(
                                                                                           className="title-header",
                                                                                           children="Option Strategy Visualizer"),
                                                                                       html.P(
                                                                                           """
                                            This app will allow you to visualize the expected profit or losses of your
                                            options strategies.
                                            """
                                                                                       ),
                                                                                       html.H6(
                                                                                           className="title-header",
                                                                                           children="Select Underlying Stock"),

                                                                                       dbc.Input(
                                                                                           id='selected_symbol',
                                                                                           placeholder='Enter Stock Symbol',
                                                                                           type='text'
                                                                                       ),
                                                                                       dbc.Button(
                                                                                           id='button-state',
                                                                                           n_clicks=0,
                                                                                           children='Get Options',
                                                                                           outline=True,
                                                                                           color='success',
                                                                                           style={
                                                                                               'marginTop': '5px'}
                                                                                       ),
                                                                                       html.Br()
                                                                                   ]
                                                                               )
                                                                           ]
                                                                       ),

                                                                   ],
                                                                   )
                                                               ],
                                                               ),
                                                       dbc.Row(children=[
                                                           dbc.Col(id='option-list-container', children=[]
                                                                   ),
                                                       ],
                                                           style={
                                                               'padding': '10px'
                                                           }),

                                                   ], width=4),

                                               dbc.Col(
                                                   # Right Pannel
                                                   dbc.Row([
                                                       dbc.Col(html.Div(id='container', children=[])),
                                                   ],
                                                       style={'padding': '10px'}
                                                   ),
                                                   width=8),
                                           ]
                                       ),
                                       dbc.Row(
                                           [
                                               dbc.Col(
                                                   html.Div(id='chart-container',
                                                            children=[]))
                                           ], style={'padding': '10px'}),

                                       dbc.Navbar(fixed='bottom',

                                                  children=
                                                  [
                                                      dbc.Row([

                                                          dbc.Col(children=[
                                                              html.Footer('Options Visuals by Francisco Romero © 2021',
                                                                          style={'textAlign': 'center'}
                                                                          )]
                                                          ),
                                                      ])
                                                  ]
                                                  ),
                                   ],
                                   style={
                                       'padding': '2.5rem'
                                   }
                                   ),

                      ],

                      style={
                          'back-ground-color': '#f2f2f2',

                      })

style_todo = {"display": "inline", "margin": "10px"}
style_done = {"textDecoration": "underline", "color": "green"}
style_done.update(style_todo)


# callback to select the underlying symbol for the first time
@app.callback(
    [
        Output(component_id='container', component_property='children'),
        Output('option-list-container', 'children')
    ],
    [
        Input(component_id='button-state', component_property='n_clicks'),
    ],
    [
        State(component_id='selected_symbol', component_property='value'),
        State(component_id='container', component_property='children')
    ],

)
# Function to select the underlying and display table
def select_chain(n_clicks, selected_symbol, new_child):
    try:
        if selected_symbol is None:
            raise PreventUpdate
        else:

            df = Ticker(selected_symbol).option_chain
            dff = df.loc[selected_symbol, df.index.get_level_values(level=1)[0].date().strftime("%Y-%m-%d"), 'calls']

            dfff = dff[['strike', 'percentChange', 'lastPrice', 'bid', 'ask', 'inTheMoney']]

            dff_display = dff[['strike', 'percentChange', 'lastPrice', 'bid', 'ask']]

            new_child = [
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id='new-div', children=[
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card(
                                        daq.ToggleSwitch(
                                            id='call-put-toggle',
                                            label=['Call', 'Put'],
                                            style={'width': '100px', 'margin': 'auto',
                                                   'color': 'green'},
                                            value=False,

                                        )),
                                    width=3
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        daq.ToggleSwitch(
                                            id='buy-sell-toggle',
                                            label=['Buy', 'Sell'],
                                            style={'width': '100px', 'margin': 'auto',
                                                   'color': 'green'},
                                            value=False,
                                        )),
                                    width=3
                                ),
                                dbc.Col(

                                    html.Div(html.P('In the Money'), style={'width': '105px',
                                                                            'backgroundColor': '#0074D9',
                                                                            'color': 'white',
                                                                            'height': '25px',
                                                                            'text-align':'center'
                                                                            })
                                    ,
                                    align='center', width=3
                                )
                            ], justify='start'),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Select(
                                        options=[{'label': i.date(), 'value': i.date()} for i in
                                                 df.index.get_level_values(level=1).unique()],
                                        value=df.index.get_level_values(level=1)[0].date(),
                                        id='dynamic-dropdown', style={
                                            'marginTop': '5px',
                                            'marginBottom': '5px'
                                        }
                                    ),
                                ),
                            ),
                            dbc.Row(
                                dbc.Col(
                                    html.Div(children=[
                                        dash_table.DataTable(
                                            id='dynamic-table',
                                            data=dfff.to_dict('records'),
                                            columns=[
                                                {'name': i, "id": i, "deletable": False, "selectable": True} for i in
                                                dff_display.columns
                                            ],
                                            style_as_list_view=True,
                                            style_cell={'padding': '5px',
                                                        'minWidth': 105, 'maxWidth': 105, 'width': 105},
                                            style_header={
                                                'backgroundColor': 'white',
                                                'fontWeight': 'bold'
                                            },
                                            style_table={'height': '500px', 'overflowY': 'auto'},
                                            fixed_rows={'headers': True},
                                            style_data_conditional=[
                                                {
                                                    'if': {
                                                        'filter_query': '{inTheMoney} contains "true"'
                                                    },
                                                    'backgroundColor': '#0074D9',
                                                    'color': 'white'
                                                }
                                            ]

                                        ),
                                    ],
                                    ),
                                ),
                            ),
                        ],
                                 ),
                    ])
                ])
            ]

            option_list_container = dbc.Card([
                dbc.CardHeader(
                    'Selected Options'
                ),

                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(id='add-state', n_clicks=0, children='Add', outline=True, color='success',
                                       style={'margin-right': '5px',
                                              'margin-bottom': '5px'}
                                       ),
                            dbc.Button(id='remove-state', n_clicks=0, children='Remove', outline=True, color='success',
                                       style={'margin-right': '5px',
                                              'margin-bottom': '5px'}
                                       ), ], )
                    ], no_gutters=True),
                    dbc.Row([
                        dbc.Col(
                            html.Div(id='option-info'))
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Div(id='option-list'))
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div('Contracts:'),
                            dbc.Input(id='contract-number', type='number', value=1, min=1, bs_size="sm", autoFocus=True)
                        ], align='center'),




                    ], justify='start', no_gutters=True),
                    dbc.Row([

                        dbc.Col([
                            html.Div(id='total-cost-placeholder', children=[],)
                        ], align='center'),
                    ], no_gutters=True, justify='start'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(id='calculate-state', n_clicks=0, children='Calculate', outline=True,
                                       color='success',
                                       style={'marginTop': '5px'})
                        ]),
                    ]),
                ])

            ])
        return new_child, option_list_container
    except:
        if selected_symbol is None:
            raise PreventUpdate
        else:
            Alert = html.Div([
                dbc.Alert([
                    html.H4("Invalid Stock Symbol", className="alert-heading"),
                    html.P(
                        "Please make sure the symbol you tried to enter is correct"
                    )], color='danger')
            ])
            return [Alert, None]


# callback to generate the the filtered. choose either put options or call options and desired expiration date
@app.callback(
    [
        Output('dynamic-table', 'data'),
    ],
    [
        Input('dynamic-dropdown', 'value'),
        Input('call-put-toggle', 'value')

    ],
    [
        State(component_id='selected_symbol', component_property='value')
    ],
    prevent_initial_call=True,

)
def update_table(expiration, put, selected_symbol):
    if not put:
        df = Ticker(selected_symbol).option_chain
        dff = df.loc[selected_symbol, expiration, 'calls']
        dfff = dff[['strike', 'percentChange', 'lastPrice', 'bid', 'ask', 'inTheMoney']]

        return [dfff.to_dict('records')]

    elif put:
        df = Ticker(selected_symbol).option_chain
        dff = df.loc[selected_symbol, expiration, 'puts']
        dfff = dff[['strike', 'percentChange', 'lastPrice', 'bid', 'ask', 'inTheMoney']]

        return [dfff.to_dict('records')]
    else:
        df = Ticker(selected_symbol).option_chain
        dff = df.loc[selected_symbol, expiration, 'calls']
        dfff = dff[['strike', 'percentChange', 'lastPrice', 'bid', 'ask', 'inTheMoney']]
        return [dfff.to_dict('records')]


@app.callback(

    [
        Output('option-info', 'children')
    ],
    [
        Input('dynamic-table', 'active_cell'),
        Input('buy-sell-toggle', 'value'),
        Input('call-put-toggle', 'value'),
        Input('dynamic-dropdown', 'value'),
    ],
    [

        State('dynamic-table', 'data'),
        State('selected_symbol', 'value'),
    ],
    prevent_initial_call=True,

)
def select_option_value(active_cell, sell, put, expiration,data, symbol):
    symbol = symbol.upper()

    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        celldata = data[row]['strike']

        if sell and put:
            out_div = ['Sell', ' ', symbol, ' ', '$', celldata, ' ', expiration, ' ', 'Put']
            return [out_div]
        if sell and (put == False):
            out_div = ['Sell', ' ', symbol, ' ', '$', celldata, ' ', expiration, ' ', 'Call']
            return [out_div]
        if (sell == False) and (put == False):
            out_div = ['Buy', ' ', symbol, ' ', '$', celldata, ' ', expiration, ' ', 'Call']
            return [out_div]
        if (sell == False) and (put == True):
            out_div = ['Buy', ' ', symbol, ' ', '$', celldata, ' ', expiration, ' ', 'Put']
            return [out_div]
    else:
        return [html.Div('No Option Selected Yet')]


@app.callback(

    [
        Output('option-list', 'children')

    ],
    [
        Input('add-state', 'n_clicks'),
        Input('remove-state', 'n_clicks'),
        Input('dynamic-table', 'active_cell')

    ],
    [
        State('option-info', 'children'),
        State({"index": ALL}, "children"),
        State({"index": ALL, "type": "done"}, "value")
    ],
    prevent_initial_call=True,

)
def edit_option_list(add, remove, active_cell, new_item, items, items_done):
    if active_cell:
        triggered = [t["prop_id"] for t in dash.callback_context.triggered]
        adding = len([1 for i in triggered if i in "add-state.n_clicks"])
        clearing = len([1 for i in triggered if i == "remove-state.n_clicks"])
        new_spec = [
            (text, done) for text, done in zip(items, items_done)
            if not (clearing and done)
        ]

        if adding and len(new_spec) < 1:
            new_spec.append((new_item, []))

        new_list = [
            html.Div([
                dcc.Checklist(
                    id={"index": i, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=done,
                    style={"display": "inline"},
                    labelStyle={"display": "inline"}
                ),
                html.Div(text, id={"index": i}, style=style_done if done else style_todo)
            ], style={"clear": "both"})
            for i, (text, done) in enumerate(new_spec)
        ]

        return [new_list]
    else:
        raise PreventUpdate

@app.callback(
    Output({"index": MATCH}, "style"),
    [Input({"index": MATCH, "type": "done"}, "value")],
    prevent_initial_call=True
)
def mark_done(done):
    return style_done if done else style_todo


@app.callback(
    Output("total-cost-placeholder", "children"),
    [
        Input({"index": ALL, "type": "done"}, "value"),
        Input("contract-number", "value")
    ],
    [
        State({"index": ALL}, "children")
    ], prevent_initial_call=True
)
def total_cost(done, n, options):
    try:
        if not done[0]:
            return ['Total Cost: ','$0']
        elif done[0]:
            symbol = options[0][2]
            # Strike Price
            strike_price = options[0][5]
            # Expiration Date
            expiration = options[0][7]
            # Option Type
            option_type = options[0][9].lower() + 's'
            oc = Ticker(symbol).option_chain
            option_data = oc.loc[symbol, expiration, option_type]
            option_cost = option_data[option_data['strike'] == strike_price]['lastPrice'][0]
            cost = n * option_cost * 100

            cost_div = ['Total Cost: ','$', "{:.2f}".format(cost)]
            return cost_div
        else:
            return ['Total Cost: ', '$0']
    except IndexError:
        raise PreventUpdate


@app.callback(
    Output("chart-container", "children"),
    [
        Input("calculate-state", "n_clicks"),
    ],
    [
        State({"index": ALL, "type": "done"}, "value"),
        State({"index": ALL}, "children"),
        State('contract-number', "value")
    ],
    prevent_initial_call=True
)
def calculate_pnl(st, done, options, n):

    if done:
        # Buy or Sell
        operation = options[0][0]
        # Symbol
        symbol = options[0][2]
        # Strike Price
        strike_price = options[0][5]
        # Expiration Date
        expiration = options[0][7]
        # Option Type
        option_type = options[0][9].lower() + 's'
        # Getting Specific option data
        oc = Ticker(symbol).option_chain
        option_data = oc.loc[symbol, expiration, option_type]
        #  List or prices for the underlying in increments
        price_list = option_data['strike']
        # Underlying price
        price_dict = Ticker(symbol).price
        current_price = price_dict[symbol]['regularMarketPrice']
        # Volatility
        iv = option_data[option_data['strike'] == strike_price]['impliedVolatility']
        implied_volatility = iv.values[0]
        # risk free interest rate 3mo T-bill rate
        risk_free_rate = 0.11
        # Time to Maturity % of the year = days/365 = exp date - current date/ 365
        option_cost = option_data[option_data['strike'] == strike_price]['lastPrice'][0]
        expiration_date = datetime.datetime.strptime(expiration, '%Y-%m-%d').date()
        current_time = datetime.date.today()

        duration = (expiration_date - current_time).days

        # Generate date columns
        date_columns = []
        step = math.ceil(duration/10)

        for dt in rrule(DAILY, dtstart=current_time, until=expiration_date, interval=step):
            if len(date_columns)< 10:
                date_columns.append(dt.strftime("%d %b %y"))

        rows = []

        for i in price_list:
            item = []
            if duration < 11:
                for j in range(duration + 1):
                    profit = ((black_scholes(i, strike_price, duration, risk_free_rate, implied_volatility, option_type,
                                         j,step) - option_cost) * 100)
                    item.append(profit)
            elif duration >10:
                for j in range(len(date_columns)):
                    profit = ((black_scholes(i, strike_price, duration, risk_free_rate, implied_volatility, option_type,
                                         j,step) - option_cost) * 100)
                    item.append(profit)
            rows.append(item)

        mydf = pd.DataFrame(rows, columns=date_columns)

        mydf = mydf[mydf.select_dtypes(include=['number']).columns] * n
        # buy or sell operation

        if operation == 'Sell':
            mydf = mydf[mydf.select_dtypes(include=['number']).columns] * (-1)
        elif operation == 'Buy':
            mydf = mydf[mydf.select_dtypes(include=['number']).columns] * (1)

        mydf['Stock_Price'] = price_list.values
        mydff = pd.melt(mydf, id_vars=['Stock_Price'], value_vars=date_columns)

        short_df = mydff[
            (mydff['Stock_Price'] < current_price * 1.2) & (mydff['Stock_Price'] > (current_price * 0.80))]
        fig = px.line(short_df, x='Stock_Price', y="value", color='variable', template='plotly_white',
                      labels={
                          "Stock_Price": "Underlying Price (USD)",
                          "value": "Gains (USD)",
                          "variable": "Date"
                      },
                      )

        fig.update_yaxes(tickprefix='$')
        fig.update_xaxes(tickprefix='$')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
        fig.add_vline(x=current_price, annotation_text='Current Price')

        new_chart = dbc.Card([

            dbc.CardBody([dcc.Graph(figure=fig)])

        ])
        print(implied_volatility)
        print(iv)
        return [new_chart]


if __name__ == '__main__':
    app.run_server(debug=True)

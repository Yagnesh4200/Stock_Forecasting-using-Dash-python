import pandas as pd
from pandas.io.formats import style
from pandas_datareader import data as pdr
from datetime import datetime as date
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from dash.exceptions import PreventUpdate
from model import predict
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
import requests

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

common_style = {'margin': '10px'}
def get_stock_price_fig(df):
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date", markers=True)
    fig.update_layout(title_x=0.5)
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode="lines+markers")
    return fig

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

common_style = {'margin': '10px'}
def get_stock_price_fig(df):
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date", markers=True)
    fig.update_layout(title_x=0.5)
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode="lines+markers")
    return fig

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(children="Stock  visualiser", className='header-title'),
        ],
            className='start',
            style={'paddingTop': '1%', 'textAlign': 'center'}
        ),
        html.Div([
            dcc.Input(id='input', type='text', placeholder='Enter stock code', className='form-control', style=common_style),
        ], className='input-section'),
        html.Div([
            html.Button('Submit', id='submit-name', n_clicks=0, className='btn btn-primary', style=common_style),
        ], className='input-section'),
        html.Div([
            html.Label('Select a date range:'),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(1995, 8, 5),
                max_date_allowed=date.now(),
                initial_visible_month=date.now(),
                end_date=date.now().date(),
                style={'fontSize': '18px', 'marginTop': '10px','marginLeft':'10px'}
            ),
            html.Div(id='output-container-date-picker-range', children='You have selected a date', className='date-info')
        ], className='date-picker-section'),
        html.Div([
            html.Button('Stock Price', id='submit-val', n_clicks=0, className='btn btn-success', style=common_style),
        ], className='button-section'),
        html.Div([
            html.Button('Indicators', id='submit-ind', n_clicks=0, className='btn btn-info', style=common_style),
        ], className='button-section'),
        html.Div([
            dcc.Input(id='Forcast_Input', type='text', placeholder='Enter days for forecast', className='form-control', style=common_style),
        ], className='button-section'),
        html.Div([
            html.Button('No of days to forecast', id='submit-forc', n_clicks=0, className='btn btn-warning', style=common_style),
        ], className='button-section')
    ], className='nav'),
    html.Div([
        html.Div([
            html.Img(id='logo'),
            html.H1(id='name')
        ],
            className="header"),
        html.Div(
           id="description",
           className="description_ticker",
           style={
               'background-color': 'black',
               'color': 'white',
               'padding': '10px',
               'border-radius': '10px',
               'margin-bottom': '10px'
           }
        ),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ], className="content")
], className="container")
import requests

@app.callback([
    Output('description', 'children'),
    Output('name', 'children'),
    Output('submit-val', 'n_clicks'),
    Output('submit-ind', 'n_clicks'),
    Output('submit-forc', 'n_clicks'),
    Input('submit-name', 'n_clicks'),
    State('input', 'value')])
def update_data(n, val):
    if n is None:
        return (
            "Hey there! Please enter a legitimate stock code to get details.",
            "Stonks",
            None,
            None,
            None
        )
    else:
        if val is None:
            raise PreventUpdate
        else:
            try:
                ticker = yf.Ticker(val)
                inf = ticker.info
                short_name = inf.get('shortName', 'Stonks')
                long_description = inf.get('longBusinessSummary', 'Description not available')

                return long_description, short_name, None, None, None
            except Exception as e:
                print(f"Error fetching data: {e}")
                return (
                    "Error fetching stock information. Please try again.",
                    "Stonks",
                    None,
                    None,
                    None
                )

@app.callback([
    Output('graphs-content','children'),
    Input('submit-val', 'n_clicks'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    State('input', 'value')])

def update_graph(n,start_date,end_date,val):
  if n == None:
        return [""]
        #raise PreventUpdate
  if val == None:
    raise PreventUpdate
  else:
    if start_date != None:
      df = yf.download(val,str( start_date) ,str( end_date ))
    else:
      df = yf.download(val)  
  df.reset_index(inplace=True)
  fig = get_stock_price_fig(df)
  return [dcc.Graph(figure=fig)]

@app.callback([Output("main-content", "children")], [
    Input("submit-ind", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("input", "value")])
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

@app.callback([
  Output("forecast-content","children"),
  Input("submit-forc","n_clicks"),
  State("Forcast_Input","value"),
  State("input","value")

])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    x=int(n_days)
    fig = predict(val, x + 1)
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run_server(debug=True)
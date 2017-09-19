import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv('data/testdata.csv')
fields = list(df.columns)

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Data embedding'),

    #html.Div(children='''
    #    Dash: A web application framework for Python.
    #'''),

    html.Label('X-axis'),
    dcc.Dropdown(
        id='x_dropdown',
        options = [{'label':v, 'value':v} for v in fields],
        #value='MTL'
    ),

    html.Label('Y-axis'),
    dcc.Dropdown(
        id='y_dropdown',
        options = [{'label':v, 'value':v} for v in fields],
        #value='MTL'
    ),

    dcc.Graph(
        id='plain-scatterplot',
        figure={
            'data': [
                go.Scatter(x=df['A'], y=df['B'], mode='markers',
                           marker=dict(size=10))
            ],
            'layout': {
                'title': 'Plot of B against A'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

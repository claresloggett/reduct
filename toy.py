
'''
toy app to explore data handling techniques.
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
#from dash.exceptions import PreventUpdate

import dash_table_experiments as dt
import flask
from flask_caching import Cache

import plotly.graph_objs as go
import plotly

import uuid
import base64
import io
import datetime
import time
import argparse
import os
import numpy as np
import pandas as pd

from ingest_data import parse_input
from transform_data import complete_missing_data, pca_transform, mds_transform, tsne_transform


app_dir = os.getcwd()

filecache_dir = os.path.join(app_dir, 'cached_files')

# Parse command-line
parser = argparse.ArgumentParser(description='Toy app')
#parser.add_argument('infile', help='CSV file of data to visualise')

# max_PCs
args = parser.parse_args()

# read and parse data
#data = pd.read_csv(args.infile)

app = dash.Dash()

#app.scripts.config.serve_locally = True

# app.server is the flask app
# in current dash version, static as path won't work?
# later also replace with app.send_from_directory?
@app.server.route('/static_files/<path:path>')
def static_file(path):
    return flask.send_from_directory(os.path.join(app_dir,'static'), path)

#app.css.config.serve_locally = True
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"})
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css"})
app.css.append_css({'external_url': '/static_files/app_layout.css'})

app.scripts.append_script({'external_url': 'http://code.jquery.com/jquery-3.3.1.min.js'})
app.scripts.append_script({'external_url': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js'})

cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    #'CACHE_TYPE': 'filesystem',
    #'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 50
})

# *** Define UI and other layout elements ***

upload_data = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )

slider = html.Div([
    html.Label("Noise: ", style={'display':'inline-block'}),
    dcc.Slider(id='slider', min=0, max=10, step=1, value=0),
])

# *** Top-level app layout ***

def serve_layout():
    print('Calling serve_layout')
    session_id = str(uuid.uuid4())
    layout =  html.Div(children=[

        html.Div(session_id, id='session-id'), #style={'display': 'none'}),
        html.Div(id='filecache_marker', style={'display': 'none'}),

        upload_data,

        html.Div(id='data-table-div'),

        slider,

        dcc.Graph(id='two-column-graph'),

        # needed to load relevant CSS/JS
        html.Div(dt.DataTable(rows=[{}]),style={'display': 'none'})
    ])
    return layout

app.layout = serve_layout


def parse_table(contents, filename):
    '''
    Parse uploaded tabular file and return dataframe.
    '''
    print('Calling parse_table')
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        raise
    print('Read dataframe:')
    print(df.columns)
    print(df)
    return df


def write_dataframe(session_id, df):
    '''
    Write dataframe to disk, for now just as CSV
    For now do not preserve or distinguish filename;
    user has one file at once.
    '''
    print('Calling write_dataframe')
    filename = os.path.join(filecache_dir, session_id)
    df.to_csv(filename, index=False)

# cache memoize this and add timestamp as input!
@cache.memoize()
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    print('Calling read_dataframe')
    filename = os.path.join(filecache_dir, session_id)
    df = pd.read_csv(filename)
    # simulate reading in big data with a delay
    print('** Reading data from disk **')
    time.sleep(5)
    return df

# if there were few slider values, we could conceivably
# call this function with them all during data load,
# especially if this could be parallelised
@cache.memoize()
def add_noise(series, slider_value):
    '''
    Add noise to column.
    '''
    # simulate complex transform with a delay
    print('** Calculating data transform **')
    time.sleep(1)
    noise = np.random.randn(len(series))*slider_value
    return series+noise

@app.callback(
    Output('filecache_marker', 'children'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('upload-data', 'last_modified')],
    [State('session-id', 'children')])
def save_file(contents, filename, last_modified, session_id):
    # write contents to file
    print('Calling save_file')
    print('New last_modified would be',last_modified)
    if contents is not None:
        print('contents is not None')
        df = parse_table(contents, filename)
        write_dataframe(session_id, df)
        return str(last_modified) # not str()?

# could remove last_modified state
# but want either it or filecache timestamp as input to read_dataframe
@app.callback(Output('data-table-div', 'children'),
              [Input('filecache_marker', 'children')],
              [State('upload-data', 'last_modified'),
               State('session-id','children')])
def update_table(filecache_marker, timestamp, session_id):
    print('Calling update_table')
    if filecache_marker is not None:
        print('filecache marker is not None')
        print('filecache marker:',filecache_marker)
        try:
            df = read_dataframe(session_id, timestamp)
        except Exception as e:
            # Show exception
            return str(e)
        output = [dt.DataTable(rows=df.to_dict('records'))]
        return output


@app.callback(Output('two-column-graph', 'figure'),
              [Input('filecache_marker', 'children'),
               Input('slider','value')],
              [State('upload-data', 'last_modified'),
               State('session-id','children')])
def update_graph(filecache_marker, slider_value, timestamp, session_id):
    ''' Plot first column against second '''
    print('Calling update_graph')
    # For now no dtype checking!
    # For now no error checking either
    if filecache_marker is None:
        raise ValueError('No data yet') # want PreventUpdate
    df = read_dataframe(session_id, timestamp)
    y_noised = add_noise(df.iloc[:,1], slider_value)
    traces = [go.Scatter(x=df.iloc[:,0], y=y_noised,
                mode='markers', marker=dict(size=10, opacity=0.7),
                text=df.index)]
    figure = {
        'data': traces,
        'layout': {
            'title': 'Graph',
            'xaxis': {'title': df.columns[0]},
            'yaxis': {'title': df.columns[1]},
            'hovermode': 'closest',
        }
    }
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)

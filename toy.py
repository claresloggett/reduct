
'''
toy app to explore data handling techniques.
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

import dash_table_experiments as dt
import flask

import plotly.graph_objs as go
import plotly

import base64
import io
import datetime
import argparse
import os
import json
import pandas as pd
from sklearn.decomposition import PCA

from ingest_data import parse_input
from transform_data import complete_missing_data, pca_transform, mds_transform, tsne_transform

app_dir = os.getcwd()

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

# *** Define UI and other layout elements ***

hidden_data = html.Div(id='hidden_data',
                           children="",
                           style={'display':'none'})

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


# No figures - will be generated by callbacks
pca_plot = dcc.Graph(id='pca_plot', animate=True)

# *** Top-level app layout ***

app.layout = html.Div(children=[

    hidden_data,

    upload_data,

    html.Div(id='output-data-upload'),

    # needed to load relevant CSS/JS
    html.Div(dt.DataTable(rows=[{}]),style={'display': 'none'})

])

def parse_table(contents, filename, date):
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
        return html.Div([
            'There was an error processing this file.'
        ])

    return [
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/datatable-experiments
        dt.DataTable(rows=df.to_dict('records'))
        ]


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents, filename, datestamp):
    if contents is not None:
        return parse_table(contents, filename, datestamp)


if __name__ == '__main__':
    app.run_server(debug=True)

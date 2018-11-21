import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

import dash_table_experiments
import flask
from flask_caching import Cache

import plotly.graph_objs as go
import plotly

import argparse
import os
import io
import errno
import json
import uuid
import base64
import pandas as pd
from sklearn.decomposition import PCA

from ingest_data import parse_input
from transform_data import complete_missing_data, pca_transform, mds_transform, tsne_transform

app_dir = os.getcwd()

filecache_dir = os.path.join(app_dir, 'cached_files')


# Parse command-line
parser = argparse.ArgumentParser(description='App for visualising high-dimensional data')
parser.add_argument('infile', help='CSV file of data to visualise')
parser.add_argument('--separator', default=',', help='separator character in tabular input')
parser.add_argument('--num-pcs', type=int, default='10', help='number of principal components to present')
parser.add_argument('--field-table', dest='show_fieldtable', action='store_true',
                     help='display FieldInfo table for info and manual selection')
parser.add_argument('--hover-sampleinfo', dest='hover_sampleinfo', action='store_true',
                     help='show sample info fields on mouseover (default is just sample ID)')
parser.add_argument('--hover-data', dest='hover_data', action='store_true',
                     help='show data values on mouseover (default is just sample ID). This can be verbose.')
parser.add_argument('--colour-by-data', dest='colour_by_data', action='store_true',
                     help='allow selection of data fields as well as sampleinfo for colouring points')

args = parser.parse_args()


# read and parse data
data, sample_info, sample_info_types, field_info = parse_input(args.infile, separator=args.separator)
fields = list(data.columns)
assert list(field_info.index) == fields

# Add missingness to field_info
field_info['MissingValues'] = data.isnull().sum()

field_info_table = field_info
field_info_table['Field'] = field_info_table.index

external_scripts = [
    'http://code.jquery.com/jquery-3.3.1.min.js',
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js'
]

# will also automatically serve assets/ folder
external_css = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css'
]

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=external_css)


# Saved files and cache

#Create save file directory if it doesn't exist
try:
    os.makedirs(filecache_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# simple cache for now: thread-safe later
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_THRESHOLD': 100}
    )


# *** Define UI and other layout elements ***

hidden_data_pca = html.Div(id='hidden_data_pca',
                           children="",
                           style={'display':'none'})

hidden_data_mds = html.Div(id='hidden_data_mds',
                           children="",
                           style={'display':'none'})

hidden_data_tsne = html.Div(id='hidden_data_tsne',
                           children="",
                           style={'display':'none'})

dummy_input = html.Div(id='dummy_input',
              children=None,
              style={'display':'none'})

if args.show_fieldtable:
    # Create the fieldinfo table for selecting fields
    fieldinfo_div = html.Div(id='fieldinfo_div',children=[
        html.Label('Include fields'),
        dash_table_experiments.DataTable(
            id='field_selector_table',
            rows=field_info_table.to_dict('records'),
            columns=['Field'] + [f for f in field_info_table.columns if f!='Field'], #put field first
            row_selectable=True,
            sortable=True,
            selected_row_indices=list(range(len(field_info_table))) #by number, not df index
        )])
else:
    fieldinfo_div = html.Div(id='fieldinfo_div')

def make_pca_dropdowns(pca_dropdown_values=[], previous_x=None, previous_y=None):
    """
    Create the children of the pca_axes_selectors div.
    """
    new_html = [
        html.Div([
            html.Label('X-axis'),
            dcc.Dropdown(
                id='x_dropdown',
                options=pca_dropdown_values,
                value=previous_x
            )]
        ),
        html.Div([
            html.Label('Y-axis'),
            dcc.Dropdown(
                id='y_dropdown',
                options=pca_dropdown_values,
                value=previous_y
            )
        ])
    ]
    return new_html




general_plot_options = html.Div(id='general_plot_options',children=[
    html.Label('Scale numeric fields', id='numericfields'),
    dcc.RadioItems(
        id='scale_selector',
        options=[{'label':"Scale numeric fields to std=1", 'value':True},
                 {'label':"Leave unscaled", 'value':False}],
        value=False  # TODO: set default to True if any categorical fields?
    ),

    html.Div(id='missing_data', children=[
        html.Label("Missing data:"),
        dcc.RadioItems(id='missing_data_selector',
            options=[{'label':"Drop fields with any missing values", 'value':'drop_fields'},
                     {'label':"Drop samples with any missing values", 'value':'drop_samples'},
                     {'label':"Fill missing values", 'value':'fill_values'}],
            value='fill_values'
        ),

        html.Div(id='missing_fill_selectors', children=[
            html.Label(children="Missing value fill in numeric fields:", id='missing_numeric_label'),
            dcc.RadioItems(id='missing_numeric_fill',
                options=[{'label':"Replace with zero", 'value':'zeroes'},
                         {'label':"Replace with mean value for field", 'value':'mean'}],
                value='mean'
            ),

            html.Label("Missing value fill in categorical fields:", id='missing_categorical_label'),
            dcc.RadioItems(id='missing_categorical_fill',
                options=[{'label':"Replace with 'Unknown'", 'value':'common_unknown'},
                         {'label':"Replace with unique category per sample",# - this can stop unknowns clustering",
                          'value':'unique_unknown'}],
                value='common_unknown'
            )
        ])
    ])
])

# initialise pca_axes_selectors with empty dropdowns; will be created by callback
pca_axes_selectors = html.Div(id='pca_axes_selectors',
                              children=make_pca_dropdowns())

colour_selector = html.Div(id='colour_selector', children=[
    html.Label('Colour points by'),
    dcc.Dropdown(
        id='colour_dropdown',
        options = [{'label':'None','value':'NONE'}],
        value='NONE'
    )
])

# this is a function so that we don't have the same component on two panes
def data_info():
    return dcc.Markdown(className='data_info',
                        children="""Viewing {0}, {2} fields, {1} samples total""".format(os.path.basename(args.infile),
                                                                                    *data.shape))

# No figures - will be generated by callbacks
pca_plot = dcc.Graph(id='pca_plot', animate=True)
mds_plot = dcc.Graph(id='mds_plot', animate=True)
tsne_plot = dcc.Graph(id='tsne_plot', animate=True)

pca_extra_stuff = html.Div(id='pca_extra_stuff',children=[
    dcc.Graph(id='pc_composition')
])

# TODO: what if the required perplexity is much bigger than 100?
# an alternate text box the user can type in would fix this
# - set it from the slider or from user input
default_perplexity = 10
tsne_controls = html.Div(id='tsne_controls',children=[
    html.Div([
        html.Label("Perplexity: {}".format(default_perplexity),
                   id='tsne_perplexity_label',
                   style={'display':'inline-block'}),
        dcc.Slider(id='tsne_perplexity_slider',
                   min=1, max=100, step=1, value=default_perplexity,
                   marks = {n:str(n) for n in [1,20,40,60,80,100]},
                   updatemode='drag'),
    ]),
    html.Button('Recalculate tSNE', id='tsne_button')
])

def define_tab_li(id, target, text, active=False):
    classes = "nav-link"
    if active:
        classes += " active"
    return html.Li(className="nav-item",
                children=[
                html.A(id=id,
                       className=classes,
                       href='#'+target,
                       children=text,
                       **{'data-toggle': 'tab'})
                ])

# *** Top-level app layout ***

def serve_layout():
    session_id = str(uuid.uuid4())
    return html.Div(children=[

            html.Div(session_id, id='session_id', style={'display': 'none'}),
            html.Div(id='filecache_timestamp', style={'display': 'none'}),

            hidden_data_pca,
            hidden_data_mds,
            hidden_data_tsne,
            dummy_input,

            html.Div(id='header_bar', children=[
                html.Div(id='app_label_box',children=[
                    html.Label('reduct', id='app_name')
                    ]),
                #  plot_type_selector
                html.Ul(id='tabs',className="nav nav-tabs",children=[
                    define_tab_li(id="upload_tab", target="upload_panel", text="Upload", active=True),
                    define_tab_li(id="pca_tab", target="pca_panel", text="PCA"),
                    define_tab_li(id="mds_tab", target="mds_panel", text="MDS"),
                    define_tab_li(id="tsne_tab", target="tsne_panel", text="tSNE")
                ])
            ]),

            html.Div(id='sidebar',children=[
                fieldinfo_div,
                general_plot_options,
                colour_selector,
                html.Div(id='lower_padding')
            ]),

            html.Div(id='main_content',className='tab-content',children=[
                html.Div(id='upload_panel', className='tab-pane active', children=[
                    html.Div(id='upload_box', children=[
                        dcc.Upload(id='upload_data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        multiple=False
                    )])
                ]),
                html.Div(id='pca_panel', className='tab-pane', children=[
                    #data_info(),
                    pca_axes_selectors,
                    pca_plot,
                    pca_extra_stuff
                ]),
                html.Div(id='mds_panel', className='tab-pane', children=[
                    #data_info(),
                    mds_plot
                ]),
                html.Div(id='tsne_panel', className='tab-pane', children=[
                    #data_info(),
                    tsne_controls,
                    tsne_plot
                ]),
            ])
        ])

app.layout = serve_layout


def write_dataframe(filename, df):
    '''
    Write dataframe to disk.
    '''
    path = os.path.join(filecache_dir, filename)
    df.to_pickle(path)

@cache.memoize()
def read_dataframe(filename, timestamp):
    '''
    Read dataframe from disk.
    '''
    path = os.path.join(filecache_dir, filename)
    df = pd.read_pickle(path)
    return df

# TODO: could allow user to specify
# TODO: could try parsing with each kind and see which works and has most columns
def guess_filetype(filename):
    extension = filename.split('.')[-1]
    if extension=='csv':
        return 'csv'
    elif extension=='tsv':
        return 'tsv'
    elif 'xls' in extension:
        return 'excel'
    else:
        print('Warning: unknown file extension; guessing CSV')

def parse_table(contents, filename):
    '''
    Parse uploaded tabular file and return dataframes
    (data, sample_info, sample_info_types, field_info).
    '''
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    filetype = guess_filetype(filename)
    if filetype in ['csv','tsv']:
        f = io.StringIO(decoded.decode('utf-8'))
    elif filetype=='excel':
        f = io.BytesIO(decoded)
    else:
        # unrecognised filetype should be caught earlier
        assert False

    try:
        data, sample_info, sample_info_types, field_info = \
            parse_input(f, filetype=filetype)
    except Exception as e:
        # TODO: show exception through browser
        print(e)
        raise

    print('Read dataframe:')
    print(data.columns)
    print(sample_info.columns)
    print(sample_info_types.columns)
    print(field_info.columns)

    return (data, sample_info, sample_info_types, field_info)

@app.callback(
    Output('filecache_timestamp', 'children'),
    [Input('upload_data', 'contents'),
     Input('upload_data', 'filename'),
     Input('upload_data', 'last_modified')],
    [State('session_id', 'children')])
def save_data(contents, filename, last_modified, session_id):
    # write contents to file
    if contents is not None:
        (data, sample_info, sample_info_types, field_info) = \
            parse_table(contents, filename)
        # We'll store objects separately for easier change of
        # storage methods later, or backwards compatibility of objects
        for (df, suffix) in zip(
                [data, sample_info, sample_info_types, field_info],
                ['data','sampleinfo','sampleinfotypes','fieldinfo']):
            write_dataframe(session_id+'_'+suffix, df)
        return last_modified


@cache.memoize()
def get_completed_data(session_id, timestamp, selected_fields, method,
                       numeric_fill, categorical_fill):
    """
    Get dataset and call complete_missing_data() to create
    dataframe with no missing values.
    Return completed dataframe.
    Memoised by completion settings, session and upload timestamp.
    """
    data = read_dataframe(session_id+'_data', timestamp)
    field_info = read_dataframe(session_id+'_fieldinfo', timestamp)

    if not args.show_fieldtable:
        assert selected_fields is None
        selected_fields = list(range(data.shape[1]))

    # TODO: we don't really need fields_kept or samples_kept
    # we could get this from the columns and index of completed
    completed, fields_kept, samples_kept = complete_missing_data(
        data.iloc[:,selected_fields],
        field_info.iloc[selected_fields,:],
        method=method, numeric_fill=numeric_fill,
        categorical_fill=categorical_fill)

    return completed

@cache.memoize()
def get_mds_data(session_id, timestamp, scale, selected_fields,
        fill_method, numeric_fill, categorical_fill):
    """
    Get completed dataset and call mds_transform.
    Return transformed data.
    Memoised by completion settings, MDS options, session and
    upload timestamp.
    """
    data = get_completed_data(session_id, timestamp, selected_fields,
        fill_method, numeric_fill, categorical_fill)
    field_info = read_dataframe(session_id+'_fieldinfo', timestamp)
    # TODO: we only really need transformed
    mds, transformed, original_fields = mds_transform(
        data, field_info.loc[data.columns,:], scale=scale)

    return transformed

# Build controls list dynamically, based on available selectors at launch
main_input_components = [Input('scale_selector','value'),
                         Input('missing_data_selector','value'),
                         Input('missing_numeric_fill','value'),
                         Input('missing_categorical_fill','value')]
main_input_components_state = [State('scale_selector','value'),
                               State('missing_data_selector','value'),
                               State('missing_numeric_fill','value'),
                               State('missing_categorical_fill','value')]
if args.show_fieldtable:
    main_input_components.append(Input('field_selector_table','selected_row_indices'))
    main_input_components_state.append(State('field_selector_table','selected_row_indices'))
else:
    main_input_components.append(Input('dummy_input','children'))
    main_input_components_state.append(State('dummy_input','children'))

@app.callback(
    Output('hidden_data_pca', 'children'),
    main_input_components
)
def update_pca(scale, missing_data_method, numeric_fill, categorical_fill, selected_fields):
    """
    Re-do the PCA based on included fields and missing data handling.
    Store in a hidden div.
    Logical order of data processing is:
    - filter out any fields the user has manually deselected
    - apply chosen missing data method
    - PCA
    """
    print("Updating PCA data")
    if not args.show_fieldtable:
        assert selected_fields is None
        selected_fields = list(range(data.shape[1]))
    data_completed, fields_kept, samples_kept = complete_missing_data(
                                         data.iloc[:,selected_fields],
                                         field_info.iloc[selected_fields,:],
                                         missing_data_method,
                                         numeric_fill, categorical_fill)
    # fields_kept is a boolean over fields_info.index[selected_fields].
    # samples_kept and fields_kept, plus selected_fields, are
    # already applied in calculating data_completed.
    # However we need to subset field_info.
    # Could reapply selected_fields and apply fields_kept,
    # or can just take data_completed.columns
    pca, transformed, components, original_fields = pca_transform(data_completed,
                                     field_info.loc[data_completed.columns,:],
                                     max_pcs=args.num_pcs,
                                     scale=scale)
    print("PCA results shape {}".format(transformed.shape))
    return json.dumps({'transformed': transformed.to_json(orient='split'),
                       'variance_ratios': list(pca.explained_variance_ratio_),
                       'components': components.to_json(),
                       'original_fields': original_fields})


# TODO: allow user to set a graph title? or a dataset title?
@app.callback(
    Output('tsne_perplexity_label', 'children'),
    [Input('tsne_perplexity_slider', 'value')]
)
def show_perplexity(perplexity):
    return 'Perplexity: {}'.format(perplexity)

@app.callback(
    Output('hidden_data_tsne', 'children'),
    [Input('tsne_button', 'n_clicks')],
    state=[State('tsne_perplexity_slider','value')] + main_input_components_state
)
def update_tsne(_n_clicks, perplexity, scale, missing_data_method, numeric_fill, categorical_fill, selected_fields):
    """
    Re-do the tSNE embedding based on tSNE parameters (perplexity),
    included fields and missing data handling.
    Store in a hidden div.
    Logical order of data processing is:
    - filter out any fields the user has manually deselected
    - apply chosen missing data method
    - tSNE
    """
    print("Updating tSNE data")
    if not args.show_fieldtable:
        assert selected_fields is None
        selected_fields = list(range(data.shape[1]))
    data_completed, fields_kept, samples_kept = complete_missing_data(
                                         data.iloc[:,selected_fields],
                                         field_info.iloc[selected_fields,:],
                                         missing_data_method,
                                         numeric_fill, categorical_fill)
    # fields_kept is a boolean over fields_info.index[selected_fields].
    # samples_kept and fields_kept, plus selected_fields, are
    # already applied in calculating data_completed.
    # However we need to subset field_info.
    # Could reapply selected_fields and apply fields_kept,
    # or can just take data_completed.columns
    tsne, transformed, original_fields = tsne_transform(data_completed,
                                     field_info.loc[data_completed.columns,:],
                                     scale=scale,
                                     perplexity=perplexity)
    print("tSNE results shape {}".format(transformed.shape))
    return json.dumps({'transformed': transformed.to_json(orient='split'),
                       'original_fields': original_fields})


@app.callback(
    Output('colour_dropdown', 'options'),
    [Input('filecache_timestamp','children')],
    state=[State('session_id', 'children')])
def update_colour_dropdown(timestamp, session_id):
    sample_info = read_dataframe(session_id+'_sampleinfo', timestamp)
    if args.colour_by_data:
        # FieldInfo is faster to get than data itself
        field_info = read_dataframe(session_id+'_fieldinfo', timestamp)
        colour_fields = [{'label':'None','value':'NONE'}] + \
                        [{'label':val,'value':'SINF'+val} for val in list(sample_info.columns)] + \
                        [{'label':val,'value':'DATA'+val} for val in list(field_info.index)]
    else:
        colour_fields = [{'label':'None','value':'NONE'}] + \
                        [{'label':val,'value':'SINF'+val} for val in list(sample_info.columns)]
    return colour_fields

@app.callback(
    Output('colour_dropdown', 'value'),
    [Input('colour_dropdown', 'options')]
)
def update_colour_dropdown_selection(_options):
    # If dropdown list ever changes, reset to no colour
    return 'NONE'


@app.callback(
    Output('pca_axes_selectors','children'),
    [Input('hidden_data_pca','children')],
    state=[State('x_dropdown','value'), State('y_dropdown','value')] # previous state
)
def update_pca_axes(transformed_data_json, previous_x, previous_y):
    """
    When PCA has been updated, re-generate the lists of available axes.
    """
    print("Updating PCA axes dropdowns")
    if transformed_data_json=="":
        print("Data not initialised yet; skipping axes callback")
        return make_pca_dropdowns()
    stored_data = json.loads(transformed_data_json)
    transformed = pd.read_json(stored_data['transformed'], orient='split')
    variance_ratios = stored_data['variance_ratios']
    pca_dropdown_values = [{'label':"{0} ({1:.3} of variance)".format(n,v), 'value':n}
                           for (n,v) in zip(transformed.columns,variance_ratios)]
    # If old selected compontents not available,
    # set x and y to PCA1 and PCA2 respectively
    if previous_x not in transformed.columns:
        previous_x = transformed.columns[0]
    if previous_y not in transformed.columns:
        previous_y = transformed.columns[1]

    return make_pca_dropdowns(pca_dropdown_values, previous_x, previous_y)

@app.callback(
    Output('pca_plot','figure'),
    [Input('x_dropdown','value'), Input('y_dropdown','value'),
     Input('colour_dropdown','value')],
    state=[State('hidden_data_pca', 'children')]
)
def update_pca_plot(x_field, y_field, colour_field_selection, stored_data):
    # If storing transformed data this way, ought to memoise PCA calculation
    print("Updating PCA figure")

    figure = create_plot(x_field=x_field,
                         y_field=y_field,
                         stored_data=stored_data,
                         colour_field_selection=colour_field_selection,
                         plot_title='PCA',
                         xaxis_label=x_field,
                         yaxis_label=y_field)

    return figure

# TODO: for now filecache_timestamp is an input
# if we want to reset missing data controls, need to callback to generate
# them, and set timestamp to input to that
@app.callback(
    Output('mds_plot','figure'),
    main_input_components +
    [Input('colour_dropdown','value')],
    state=[State('session_id','children'),
           State('filecache_timestamp','children')]
)
def update_mds_plot(scale, missing_data_method, numeric_fill, categorical_fill,
                    selected_fields, colour_field_selection,
                    session_id, timestamp):
    print("Updating MDS figure")
    print("Colour field:",colour_field_selection)

    transformed = get_mds_data(
        session_id, timestamp, scale, selected_fields,
        missing_data_method, numeric_fill, categorical_fill)

    # TODO: we are reading and passing entire original data which is only used if hover_data
    data = read_dataframe(session_id + '_data', timestamp)
    field_info = read_dataframe(session_id + '_fieldinfo', timestamp)
    sample_info = read_dataframe(session_id + '_sampleinfo', timestamp)
    sample_info_types = read_dataframe(session_id + '_sampleinfotypes', timestamp)

    figure = create_plot(x_field='MDS dim A',
                         y_field='MDS dim B',
                         transformed=transformed,
                         data=data,
                         sample_info=sample_info,
                         sample_info_types=sample_info_types,
                         field_info=field_info,
                         colour_field_selection=colour_field_selection,
                         plot_title='MDS',
                         xaxis_label='MDS dim A',
                         yaxis_label='MDS dim B')

    return figure

@app.callback(
    Output('tsne_plot','figure'),
    [Input('hidden_data_tsne', 'children'), Input('colour_dropdown','value')]
)
def update_tsne_plot(stored_data, colour_field_selection):
    # If storing transformed data this way, ought to memoise calculation
    print("Updating tSNE figure")

    figure = create_plot(x_field='A',
                         y_field='B',
                         stored_data=stored_data,
                         colour_field_selection=colour_field_selection,
                         plot_title='tSNE',
                         xaxis_label='tSNE dim A',
                         yaxis_label='tSNE dim B')

    return figure

def create_plot(x_field, y_field, transformed, data,
                sample_info, sample_info_types, field_info,
                colour_field_selection,
                plot_title, xaxis_label, yaxis_label):
    """
    Create a scatter plot based on already-transformed data.
    Returns the figure.
    """
    print("Plotting {} points".format(len(transformed)))
    # In case we dropped any samples during transformation
    sample_info_used = sample_info.loc[transformed.index,:]

    # Show sample ID on hover
    hover_text = transformed.index
    if args.hover_sampleinfo:
        # Show sample info fields on hover
        hover_text = hover_text.str.cat([sample_info_used[field].apply(lambda v:"{}={}".format(field,v))
                                         for field in sample_info_used.columns],
                                         sep=' | ')
    if args.hover_data:
        # Show data values on hover. Will include deselected fields and filtered fields.
        data_used = data.loc[transformed.index,:]
        hover_text = hover_text.str.cat([data_used[field].apply(lambda v:"{}={}".format(field,v))
                                         for field in data_used.columns],
                                         sep=' | ')

    colour_field_source, colour_field = colour_field_selection[:4], colour_field_selection[4:]
    if colour_field_source == 'NONE':
        # No colouring
        traces = [go.Scatter(x=transformed[x_field], y=transformed[y_field],
                  mode='markers', marker=dict(size=10, opacity=0.7),
                  text=hover_text)]
    else:
        # Colour by colour field
        if colour_field_source=='SINF':
            colour_values = sample_info_used[colour_field]
            colour_field_type = sample_info_types.loc[colour_field,'InfoType']
        else:
            assert colour_field_source=='DATA'
            colour_values = data.loc[transformed.index,colour_field]
            colour_field_type = field_info.loc[colour_field,'FieldType']
        # Use continuous colour scale if Numeric, and discrete if Categorical
        if colour_field_type=='Numeric':
            #colour_values[ colour_values.isnull() ] = 0  # better to let plotly handle
            traces = [go.Scatter(x=transformed[x_field], y=transformed[y_field],
                      mode='markers',
                      marker=dict(size=10, opacity=0.7,
                                  color=colour_values,
                                  showscale=True),
                      text=hover_text)]
        else:
            # Treat colour as a categorical field
            # Make separate traces to get colours and a legend
            assert colour_field_type in ['Categorical','OrderedCategorical']
            traces = []
            # points with missing values
            if colour_values.isnull().sum() > 0:
                rows = colour_values.isnull()
                traces.append(go.Scatter(x=transformed.loc[rows,x_field], y=transformed.loc[rows,y_field],
                              mode='markers', marker=dict(size=10, opacity=0.7),
                              name='Unknown', text=hover_text[rows]))
            # points with a colour field value - in category order if pandas category, else sorted
            try:
                unique_colour_values = colour_values.cat.categories
            except AttributeError:  # no .cat accessor, not categorical
                unique_colour_values = sorted(colour_values.unique(), key=lambda x:str(x))
            for value in unique_colour_values:
                rows = colour_values == value
                traces.append(go.Scatter(x=transformed.loc[rows,x_field], y=transformed.loc[rows,y_field],
                              mode='markers', marker=dict(size=10, opacity=0.7),
                              name=value, text=hover_text[rows]))

    figure = {
        'data': traces,
        'layout': {
            'title': plot_title,
            'xaxis': {'title': xaxis_label},
            'yaxis': {'title': yaxis_label},
            'hovermode': 'closest',
        }
    }
    return figure


@app.callback(
    Output('pc_composition','figure'),
    [Input('x_dropdown','value'), Input('y_dropdown','value')],
    state=[State('hidden_data_pca', 'children')]
)
def update_pc_composition(x_field, y_field, stored_data):
    print("Updating PC composition graph")
    if stored_data=="":
        print("Data not initialised yet; skipping PC composition callback")
        return {'data': [], 'layout': {'title': 'Calculating plot...'}}
    if x_field is None or y_field is None:
        print("Axes dropdowns not initialised yet; skipping PC composition callback")
        return {'data': [], 'layout': {'title': 'Calculating plot...'}}
    stored_json = json.loads(stored_data)
    components = pd.read_json(stored_json['components'])
    original_fields = stored_json['original_fields']
    pcx = components[x_field].pow(2)
    pcy = components[y_field].pow(2)

    original_fieldlist = list(set(original_fields.values()))
    pcx_original = pd.Series(0, index=original_fieldlist,
                                name=pcx.name+'_originalfields')

    # Calculate total contribution from each original non-encoded field
    # TODO: Is there a faster way?
    for (field, sqvalue) in pcx.items():
        pcx_original.loc[original_fields[field]] += sqvalue
    pcy_original = pd.Series(0, index=original_fieldlist)
    for (field, sqvalue) in pcy.items():
        pcy_original.loc[original_fields[field]] += sqvalue

    xlabels,xsizes = zip(*[(field, sqvalue)
                      for (field, sqvalue)
                      in pcx_original.sort_values(ascending=False)[:5].items()
                      if sqvalue > 0.01][::-1])
    ylabels,ysizes = zip(*[(field, sqvalue)
                      for (field, sqvalue)
                      in pcy_original.sort_values(ascending=False)[:5].items()
                      if sqvalue > 0.01][::-1])

    hovertext_x = ["; ".join(["{}={}".format(name,value) for (name,value) in field_info.loc[field,:].items()])
                      for field in xlabels]
    hovertext_y = ["; ".join(["{}={}".format(name,value) for (name,value) in field_info.loc[field,:].items()])
                      for field in ylabels]

    x_bargraph = go.Bar(y=xlabels, x=xsizes,
                        text=hovertext_x,
                        orientation='h', width=0.6,
                        marker={'color':'lightblue'})
    y_bargraph = go.Bar(y=ylabels, x=ysizes,
                        text=hovertext_y,
                        orientation='h', width=0.6,
                        marker={'color':'lightblue'})

    pc_graphs = plotly.tools.make_subplots(cols=2, subplot_titles=[x_field,y_field])
    pc_graphs.append_trace(x_bargraph, 1, 1)
    pc_graphs.append_trace(y_bargraph, 1, 2)
    pc_graphs['layout']['xaxis1'].update(range=[0,1])
    pc_graphs['layout']['xaxis2'].update(range=[0,1])
    pc_graphs['layout']['yaxis1'].update(showline=True, mirror=True)
    pc_graphs['layout']['yaxis2'].update(showline=True, mirror=True)
    pc_graphs['layout'].update(title='Principal component approx composition',
                               showlegend=False,
                               height=300)

    return pc_graphs


@app.callback(
    Output('missing_fill_selectors','style'),
    [Input('missing_data_selector','value')],
)
def grey_fill_dropdowns(missing_data_method):
    """Grey/ungrey fill radio elements when they are being ignored/not ignored."""
    if missing_data_method=='fill_values':
        return {}
    else:
        return {'color': 'gray'}


if __name__ == '__main__':
    app.run_server(debug=True)

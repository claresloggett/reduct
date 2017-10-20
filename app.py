import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

import dash_table_experiments

import plotly.graph_objs as go

import argparse
import os.path
import json
import pandas as pd
from sklearn.decomposition import PCA

from ingest_data import parse_input
from transform_data import complete_missing_data, pca_transform

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

# max_PCs
args = parser.parse_args()

# read and parse data
data, sample_info, sample_info_types, field_info = parse_input(args.infile, separator=args.separator)
fields = list(data.columns)
assert list(field_info.index) == fields

# Add missingness to field_info
field_info['MissingValues'] = data.isnull().sum()

field_info_table = field_info
field_info_table['Field'] = field_info_table.index

app = dash.Dash()
#app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if args.show_fieldtable:
    # Create the fieldinfo table for selecting fields
    fieldinfo_elements = [
        html.Label('Include fields'),
        dash_table_experiments.DataTable(
            id='field_selector_table',
            rows=field_info_table.to_dict('records'),
            columns=['Field'] + [f for f in field_info_table.columns if f!='Field'], #put field first
            row_selectable=True,
            sortable=True,
            selected_row_indices=list(range(len(field_info_table))) #by number, not df index
        )]
else:
    fieldinfo_elements = []

starting_axes_dropdowns = [
    html.Label('X-axis'),
    dcc.Dropdown(
        id='x_dropdown',
        value=None
        # options and value created by callback
    ),
    html.Label('Y-axis'),
    dcc.Dropdown(
        id='y_dropdown',
        value=None
        # options and value created by callback
    )]

if args.colour_by_data:
    colour_fields = [{'label':'None','value':'NONE'}] + \
                    [{'label':val,'value':'SINF'+val} for val in list(sample_info.columns)] + \
                    [{'label':val,'value':'DATA'+val} for val in list(data.columns)]
else:
    colour_fields = [{'label':'None','value':'NONE'}] + \
                    [{'label':val,'value':'SINF'+val} for val in list(sample_info.columns)]

app.layout = html.Div(children=[
    #html.H1(children='Data embedding'),

    dcc.Markdown(id='data_info',
        children="""File {0}, {2} fields, {1} samples total""".format(os.path.basename(args.infile),
                                                                         *data.shape)
    ),

    *fieldinfo_elements,

    # children will be overwritten with stored data
    html.Div(id='hidden_data_div',
             children="",
             style={'display':'none'}),

    # Dummy input
    html.Div(id='dummy_input',
             children=None,
             style={'display':'none'}),

    html.Label('Scale numeric fields'),
    dcc.RadioItems(
        id='scale_selector',
        options=[{'label':"Scale numeric fields to std=1", 'value':True},
                 {'label':"Leave unscaled", 'value':False}],
        value=False  # TODO: set default to True if any categorical fields?
    ),

    html.Div(id='missing_data',
    children=[
        html.Label("Missing data:"),
        dcc.RadioItems(id='missing_data_selector',
            options=[{'label':"Drop fields with any missing values", 'value':'drop_fields'},
                     {'label':"Drop samples with any missing values", 'value':'drop_samples'},
                     {'label':"Fill missing values", 'value':'fill_values'}],
            value='fill_values'
        ),

        html.Div(id='missing_fill_selectors',
            children=[
            html.Label(children="Missing value fill in numeric fields:", id='missing_numeric_label'),
            dcc.RadioItems(id='missing_numeric_fill',
                options=[{'label':"Replace with zero", 'value':'zeroes'},
                         {'label':"Replace with mean value for field", 'value':'mean'}],
                value='mean'
            ),

            html.Label("Missing value fill in categorical fields:", id='missing_categorical_label'),
            dcc.RadioItems(id='missing_categorical_fill',
                options=[{'label':"Replace with 'Unknown'", 'value':'common_unknown'},
                         {'label':"Replace with unique category per sample - this can stop unknowns clustering",
                          'value':'unique_unknown'}],
                value='common_unknown'
            )]
        ),
    ]),

    html.Div(id='axis_component_selectors',
             children=starting_axes_dropdowns
    ),

    html.Label('Colour points by'),
    dcc.Dropdown(
        id='colour_dropdown',
        options = colour_fields,
        value='NONE'
    ),

    dcc.Graph(
        id='pca_plot',  # No figure - will be generated by callback
        animate=True
    )

])

# Build input list for update_pca dynamically, based on available selectors
pca_input_components = [Input('scale_selector','value'),
                        Input('missing_data_selector','value'),
                        Input('missing_numeric_fill','value'),
                        Input('missing_categorical_fill','value')]
if args.show_fieldtable:
    pca_input_components.append(Input('field_selector_table','selected_row_indices'))
else:
    pca_input_components.append(Input('dummy_input','children'))

@app.callback(
    Output('hidden_data_div', 'children'),
    pca_input_components
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
    pca, transformed = pca_transform(data_completed,
                                     field_info.loc[data_completed.columns,:],
                                     max_pcs=args.num_pcs,
                                     scale=scale)
    print("PCA results shape {}".format(transformed.shape))
    return json.dumps({'transformed': transformed.to_json(orient='split'),
                       'variance_ratios': list(pca.explained_variance_ratio_)})

@app.callback(
    Output('axis_component_selectors','children'),
    [Input('hidden_data_div','children')],
    state=[State('x_dropdown','value'), State('y_dropdown','value')]
)
def update_pca_axes(transformed_data_json, previous_x, previous_y):
    """
    When PCA has been updated, re-generate the lists of available axes.
    """
    print("Updating PCA axes dropdowns")
    if transformed_data_json=="":
        print("Data not initialised yet; skipping axes callback")
        return starting_axes_dropdowns
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
    new_html = [
        html.Label('X-axis'),
        dcc.Dropdown(
            id='x_dropdown',
            options=pca_dropdown_values,
            value=previous_x
        ),

        html.Label('Y-axis'),
        dcc.Dropdown(
            id='y_dropdown',
            options=pca_dropdown_values,
            value=previous_y
        )]
    return new_html

@app.callback(
    Output('pca_plot','figure'),
    [Input('x_dropdown','value'), Input('y_dropdown','value'),
     Input('colour_dropdown','value')],
    state=[State('hidden_data_div', 'children')]
)
def update_figure(x_field, y_field, colour_field_selection, stored_data):
    # If storing transformed data this way, ought to memoise PCA calculation
    print("Updating figure")
    # Don't try to calculate plot if UI controls not initialised yet
    # Note that we must however return a valid figure specification
    if stored_data=="":
        print("Data not initialised yet; skipping figure callback")
        return {'data': [], 'layout': {'title': 'Calculating plot...'}}
    if x_field is None or y_field is None:
        print("Axes dropdowns not initialised yet; skipping figure callback")
        return {'data': [], 'layout': {'title': 'Calculating plot...'}}
    transformed = pd.read_json(json.loads(stored_data)['transformed'], orient='split')
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
            if pd.core.common.is_categorical_dtype(colour_values):
                unique_colour_values = colour_values.cat.categories
            else:
                unique_colour_values = sorted(colour_values.unique(), key=lambda x:str(x))
            for value in unique_colour_values:
                rows = colour_values == value
                traces.append(go.Scatter(x=transformed.loc[rows,x_field], y=transformed.loc[rows,y_field],
                              mode='markers', marker=dict(size=10, opacity=0.7),
                              name=value, text=hover_text[rows]))

    figure = {
        'data': traces,
        'layout': {
            'title': 'PCA',
            'xaxis': {'title': x_field},
            'yaxis': {'title': y_field},
            'hovermode': 'closest',
        }
    }
    return figure

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

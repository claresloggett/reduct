
Reduct
======

An app for visualising high-dimensional data via interactive embedding.
Uses Plotly Dash.

In experimental development.

To install after cloning this repository, create an environment and `pip install -r requirements.txt`

There is alternatively a conda environment available with versions pinned, for use in development: `environment-dev.yml`.

The simplest usage is to launch reduct on the command line like so:

```python reduct.py```

and point your browser to `127.0.0.1:8050`.

See other usage options with

```python reduct.py -h```

## Data format

Reduct reads standard CSV files. The separator can optionally be specified with `--separator`.
The first row will be treated as the header row and used to specify variable names.

By default, all columns will be treated as data columns and included in the
dimensionality reduction. By default, types will be inferred by Pandas' type logic.

Specifiers can optionally be added to variable names in the header row using a `:` suffix,
e.g. `variablename:M`. Recognised specifiers are:

* `I` - index column. Values will be treated as identifiers for points in the plot and will not be included as data in dimensionality reduction transforms. Currently, if you specify more than one index column, only the first
will be used; the rest will be silently ignored.
* `M` - metadata column. Metadata fields will *not* be included in the dimensionality reduction.
They can be used to colour points in the plot.
* `Q` - variable type is quantitative, i.e. numerical.
* `N` - variable type is nominal, i.e. categorical or factor.

This syntax is modelled after [Altair](https://altair-viz.github.io/) type specifiers.

Types will be deduced where not specified, so in general you only need to specify the
types of columns that are not obvious from the data.

Any compatible specifiers can be combined, e.g. `variablename:MN` denotes a nominal
metadata variable.

Datetimes and ordinals (ordered categoricals) are not yet handled, but are planned
for a future release.

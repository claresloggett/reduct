
import pandas as pd

# Not yet implemented:
# - field info other than FieldType(ColumnType)
# - doing anything with RowID
# - Categoricals, at all
# - auto-detection of type when there is no ColumnType row

def parse_input(infile, separator):
    """
    Parse input data from CSV file.
    Split sample and field info from actual data,
    and return (data, sample_info, field_info).
    The first row must be the header row, with column names.
    RowType is considered a reserved column name.
    Recognised RowType values are ColumnType, FieldInfo, Data.
    If ColumnType row exists, sample ID and info columns are possible,
    otherwise all columns, other than RowType, are data.
    Recognised ColumnType values are RowID, RowInfo, Numerical, Categorical,
    and OrderedCategorical.
    """
    df = pd.read_csv(infile, sep=separator, header=0)

    # Check for RowType column
    if 'RowType' in df.columns:
        data_rows = df.index[df['RowType']=='Data']
        fieldinfo_rows = df.index[df['RowType']=='FieldInfo']
        columntype_row = df.index[df['RowType']=='ColumnType']
    else:
        print("No RowType column in input data; assuming all rows are data.")
        data_rows = df.index
        fieldinfo_rows = []
        columntype_row = []

    # Check for ColumnType row
    if len(columntype_row) > 1:
        raise ValueError("More than one ColumnType row in input data.")
    elif len(columntype_row) == 0:
        print("No ColumnType column in input data; assuming all columns data fields, unless named RowType.")
        data_columns = df.columns[df.columns != 'RowType']
        # TODO: try to auto-detect column type. For now, assume numeric.
        columntypes = pd.Series(index=df.columns)
        columntypes[data_columns] = 'Numeric'
    else:
        assert len(columntype_row) == 1
        columntypes = df.loc[columntype_row,:].iloc[0]

    # Extract data and metadata

    # TODO: handle other column types
    data_columns = df.columns[columntypes.isin(['Numeric'])]

    data = df.loc[data_rows, data_columns]

    field_info = pd.DataFrame(index=data_columns)
    field_info['FieldType'] = columntypes[data_columns]

    sample_info = pd.DataFrame(index=data_rows)
    sampleinfo_columns = (columntypes == 'RowInfo')
    sample_info = df.loc[data_rows, sampleinfo_columns]

    if data.shape[0]==0:
        raise ValueError("No data rows found.")
    if data.shape[1]==0:
        raise ValueError("No data fields found.")

    print("Data shape {}".format(data.shape))
    print("Data fields: {}".format(','.join(data.columns)))
    if sample_info is not None:
        print("Sample info fields: {}".format(','.join(sample_info.columns)))

    return (data, sample_info, field_info)

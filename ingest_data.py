
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

    if 'RowType' in df.columns:
        data_rows = df['RowType']=='Data'
        fieldinfo_rows = df['RowType']=='FieldInfo'
        # data columns must at least exclude RowType column
        # it can be further restricted later
        data_columns = df.columns != 'RowType'
        columntype_row = df.index[df['RowType']=='ColumnType']
        if len(columntype_row) > 1:
            raise ValueError("More than one ColumnType row in input data.")
        if len(columntype_row) > 0:
            columntypes = df.loc[columntype_row,:].iloc[0]
            # Only handle numeric for now
            # Can override previous data_columns; RowType should not be Numeric etc
            data_columns = columntypes.isin(['Numeric'])
            data = df.loc[data_rows, data_columns]
            fields = list(df.columns[data_columns])
            field_info = pd.DataFrame(index=fields)
            field_info['FieldType'] = columntypes[data_columns]
            sample_info = pd.DataFrame(index=df.index[data_rows])
            sampleinfo_columns = (columntypes == 'RowInfo')
            sample_info = df.loc[data_rows, sampleinfo_columns]
        else:
            print("No ColumnType column in input data; assuming all columns are numeric fields.")
            data = df.loc[data_rows,data_columns]
            sample_info = None
            fields = list(df.columns[data_columns])
            field_info = pd.DataFrame(index=fields)
            field_info['FieldType'] = ['Numeric']*len(field_info)
    else:
        print("No RowType column in input data; assuming all rows are data and all columns are numeric fields.")
        data = df.loc[:,df.columns]
        sample_info = None
        fields = list(df.columns)
        field_info = pd.DataFrame(index=fields)
        field_info['FieldType'] = ['Numeric']*len(field_info)
    if data.shape[0]==0:
        raise ValueError("No data rows found.")
    if data.shape[1]==0:
        raise ValueError("No data fields found.")
    print("Data shape {}".format(data.shape))
    print("Data fields: {}".format(','.join(data.columns)))
    if sample_info is not None:
        print("Sample info fields: {}".format(','.join(sample_info.columns)))
    return (data, sample_info, field_info)

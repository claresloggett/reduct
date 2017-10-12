
import pandas as pd

def parse_input(infile, separator):
    """
    Parse input data from CSV file.
    Split sample and field info from actual data,
    and return (data, sample_info, field_info).
    The first row must be the header row, with column names.
    RowType is considered a reserved column name.
    Recognised RowType values are ColumnType, DataType, FieldInfo, Data.
    If RowType column does not exist, all rows are assumed to be Data, except
    the header (first) row.
    Recognised ColumnType values are RowID, RowInfo, and Data.
    If ColumnType row does not exist, all columns are assumed to be Data, except
    the RowType column.
    Recognised DataType values are Numeric, Categorical,
    and (not yet implemented) OrderedCategorical.
    If DataType row does not exist, all columns are assumed to be Numeric, except
    the RowType column.
    Both FieldInfo and Data columns can have DataTypes specified.
    """
    print("Reading "+infile)
    df = pd.read_csv(infile, sep=separator, header=0, dtype=object)

    print("Checking for master columns/rows")

    # Check for RowType column and get special rows
    if 'RowType' in df.columns:
        data_rows = df.index[df['RowType']=='Data']
        fieldinfo_rows = df.index[df['RowType']=='FieldInfo']
        columntype_row = df.index[df['RowType']=='ColumnType']
        datatype_row = df.index[df['RowType']=='DataType']
    else:
        print("No RowType column in input data; assuming all rows are data.")
        data_rows = df.index
        fieldinfo_rows = []
        columntype_row = []
        datatype_row = []

    # Check for ColumnType row
    if len(columntype_row) > 1:
        raise ValueError("More than one ColumnType row in input data.")
    elif len(columntype_row) == 0:
        print("No ColumnType row in input data; assuming all columns data fields, unless named RowType.")
        data_columnspec = df.columns != 'RowType'
        columntypes = pd.Series(index=df.columns)
        columntypes[data_columnspec] = 'Data'
        #data_columns = df.columns[data_columnspec]
    else:
        # There is a ColumnType row
        assert len(columntype_row) == 1
        columntypes = df.loc[columntype_row,:].iloc[0]
        data_columnspec = columntypes == 'Data'
        #data_columns = df.columns[data_columnspec]

    # Check for DataType row
    if len(datatype_row) > 1:
        raise ValueError("More than one DataType row in input data.")
    elif len(datatype_row) == 0:
        print("No DataType row in input data; assuming all data columns are numeric.")
        # TODO: try to auto-detect column type. For now, assume numeric.
        datatypes = pd.Series(index=df.columns)
        datatypes[data_columnspec] = 'Numeric'
    else:
        # There is a DataType row
        assert len(datatype_row) == 1
        datatypes = df.loc[datatype_row,:].iloc[0]

    # Check for RowID column
    if 'RowID' in columntypes.values:
        id_columns = df.columns[columntypes == 'RowID']
        if len(id_columns) > 1:
            raise ValueError("More than one RowID column found.")
        id_column = id_columns[0]
        sample_ids = list(df.loc[data_rows,id_column])
        fieldinfo_ids = list(df.loc[fieldinfo_rows,id_column])
    else:
        sample_ids = ["sample{}".format(n+1) for n in range(len(data_rows))]
        fieldinfo_ids = ["fieldinfo{}".format(n+1) for n in range(len(fieldinfo_rows))]

    # Warn the user if they have assigned datatypes to any columns that won't
    # take them.
    # i.e. if datatypes has a value, and columntypes has a value other than
    # RowInfo, Data, or ColumnType ('ColumnType' is in the RowType column, populated by 'DataType')
    ignored_datatypes = ~datatypes.isnull() & \
                           ~columntypes.isnull() & \
                           ~columntypes.isin(['RowInfo','Data','ColumnType'])
    if ignored_datatypes.sum() > 0:
        print("Warning: DataType value assigned to columns that will ignore it: " + \
                ",".join(df.columns[ignored_datatypes]))

    # Check that sample_info and data columns have recognised data types
    # If they don't, ignore them and warn the user
    # TODO: handle categorical category specifications, and ordered categories
    recognised_datatypes = datatypes.isin(['Numeric', 'Categorical'])
    unrecognised_data = data_columnspec & ~recognised_datatypes
    if unrecognised_data.sum() > 0:
        print("Warning: some data columns have unrecognised DataTypes, these will be ignored: " + \
                ",".join(df.columns[unrecognised_data]))
        data_columnspec = data_columnspec & recognised_datatypes

    data_columns = df.columns[data_columnspec]

    # Extract data, field info and sample info
    print("Extracting data")

    data = df.loc[data_rows, data_columns]

    print("Extracting field info")

    field_info = pd.DataFrame(index=data_columns)
    field_info['FieldType'] = datatypes[data_columns]
    for (fieldinfo_name, row_index) in zip(fieldinfo_ids, fieldinfo_rows):
        field_info[fieldinfo_name] = df.loc[row_index, data_columns].transpose()

    print("Extracting sample info")

    sample_info = pd.DataFrame(index=data_rows)
    sampleinfo_columns = (columntypes == 'RowInfo')
    unrecognised_sampleinfo = sampleinfo_columns & ~recognised_datatypes
    if unrecognised_sampleinfo.sum() > 0:
        print("Warning: some RowInfo columns have unrecognised DataTypes, these will be ignored: " + \
                ",".join(df.columns[unrecognised_sampleinfo]))
        sampleinfo_columns = sampleinfo_columns & recognised_datatypes
    sample_info = df.loc[data_rows, sampleinfo_columns]

    # Store datatypes types of RowInfo fields, if supplied
    sample_info_types = pd.DataFrame(datatypes[sampleinfo_columns])
    assert sample_info_types.shape[1]==1
    sample_info_types.columns = ['InfoType']

    print("Setting data column types")

    # Give numeric columns dtype of float
    # For speed, convert all numeric columns, glue back together,
    # then reorder to original order to match field_info and to match
    # user-specified input order
    is_numeric = field_info['FieldType']=='Numeric'
    data = pd.merge(data.loc[:,~is_numeric], data.loc[:,is_numeric].astype('float'),
                   left_index=True, right_index=True)[list(field_info.index)]
    # These methods are slow:
    #for field in field_info.index:
    #    if field_info.loc[field,'FieldType']=='Numeric':
    #        data[field] = pd.to_numeric(data[field])
    #
    #numeric_columns = data.columns[field_info['FieldType']=='Numeric']
    #data[numeric_columns] = data[numeric_columns].astype('float')

    # TODO: Give categorical columns appropriate dtype (pandas categorical)

    # Use sample IDs as index
    data.index = sample_ids
    sample_info.index = sample_ids

    if data.shape[0]==0:
        raise ValueError("No data rows found.")
    if data.shape[1]==0:
        raise ValueError("No data fields found.")

    print("Data shape {}, sample info shape {}, field info shape {}".format(data.shape,
                                                                            sample_info.shape,
                                                                            field_info.shape))
    #print("Data fields: {}".format(','.join(data.columns)))
    print("Sample info fields: {}".format(','.join(sample_info.columns)))
    print("Field info fields: {}".format(','.join(field_info.columns)))

    return (data, sample_info, sample_info_types, field_info)

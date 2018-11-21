
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import _get_dtype_from_object

def matches_dtypes(df, dtypes):
    """
    Return Series that is True where dtype of column matches given dtypes.
    Returns a Series with index matching df.columns.
    dtypes must be an iterable of type specifications.
    Copies match logic from pandas.DataFrame.select_dtypes();
    use same type specifications:
    * To select all *numeric* types use the numpy dtype ``numpy.number``
    * To select strings you must use the ``object`` dtype, but note that
      this will return *all* object dtype columns
    * See the `numpy dtype hierarchy
      <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`__
    * To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
    * To select timedeltas, use np.timedelta64, 'timedelta' or
      'timedelta64'
    * To select Pandas categorical dtypes, use 'category'
    * To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0),
      or a 'datetime64[ns, tz]' string
    """
    dtypes = list(map(_get_dtype_from_object, dtypes))
    boolean_list = [any([issubclass(coltype.type,t) for t in dtypes])
                     for (column,coltype) in df.dtypes.iteritems()]
    return pd.Series(boolean_list, index=df.columns)

# Guess unknown datatypes
# For now, if it's not numeric, it's categorical
# For now, just use pandas' guess at dtype
def guess_datatypes(df, known_datatypes=None):
    """
    Where known_datatypes is '', fill in with guessed datatypes.
    Return Series of resulting datatypes.
    Will be identical to known_datatypes if all datatypes were specified.
    """
    ALLOWED_VALUES = {'N','Q',''}
    if known_datatypes is None:
        known_datatypes = ['']*df.shape[1]
    if len(set(known_datatypes) - ALLOWED_VALUES) > 0:
        raise ValueError("Unrecognised datatypes: {}".format(set(known_datatypes) - ALLOWED_VALUES))

    datatypes = pd.Series(known_datatypes, index=df.columns)
    unknown = [t=='' for t in known_datatypes]
    looks_numeric = matches_dtypes(df, [np.number])
    # for now either numeric or categorical
    looks_categorical = ~looks_numeric
    datatypes[unknown & looks_numeric] = 'Q'
    datatypes[unknown & looks_categorical] = 'N'
    return datatypes

def split_typespec(colname):
    '''Split column name into fieldname and typespec'''
    if ':' in colname:
        pieces = colname.split(':')
        return (':'.join(pieces[:-1]),pieces[-1])
    else:
        return (colname,'')

def extract_specs(spec, codes="NQ", assert_single=True):
    '''Extract only certain single-letter codes from a field'''
    values = [v for v in spec if v in codes]
    if assert_single and len(values)>1:
        raise ValueError("More than one type specifier supplied to field: {}".format(spec))
    if len(values)==0:
        return ''
    else:
        return values[0]

def parse_input(infile, separator=None, filetype='csv'):
    """
    Parse input data from CSV file.
    infile is passed to Pandas and may be a filename or file handle.
    Split sample and field info from actual data.
    The first row must be the header row, with column names.
    Suffixes to variable names, split by : , will be interpreted
    as datatypes, altair-style.
    Recognised codes are:
      N : Nominal
      Q : Quantitative
    (datetimes, ordinal datatypes TBD)
    A code may also be added for the role of the field.
    Recognised codes are:
      I : Identifier - will not be treated as data for dim reducion,
                       and will be displayed by default on mouseover
    (M TBD for both fields and rows)

    Return (data, sample_info, sample_info_types, field_info).
    """
    if filetype=='csv':
        df = pd.read_csv(infile, sep=separator, header=0)
    elif filetype=='tsv':
        # or could set separator
        df = pd.read_table(infile, sep=separator, header=0)
    elif filetype=='excel':
        df = pd.read_excel(infile, header=0)
    else:
        raise ValueError(
            'Unrecognised file type {} passed to parse_input'.format(filetype))

    fieldnames, typespecs = zip(*[split_typespec(col) for col in df.columns])
    typespecs = pd.Series(typespecs, index=fieldnames)
    df.columns = fieldnames
    #print('typespecs',typespecs)
    is_index = typespecs.str.contains('I')
    is_metadata = typespecs.str.contains('M')
    is_data = ~(is_index | is_metadata)
    #print(is_index,is_metadata,is_data)

    data_rows = df.index
    data_columns = df.columns[is_data]

    print("Checking for index and metadata")

    # Field types
    specified_types = typespecs.apply(extract_specs)
    columntypes = guess_datatypes(df, known_datatypes=specified_types)

    if is_index.sum()==0:
        sample_ids = ["sample{}".format(n+1) for n in range(len(data_rows))]
        #fieldinfo_ids = ["fieldinfo{}".format(n+1) for n in range(len(fieldinfo_rows))]
    elif is_index.sum()>1:
            raise ValueError("More than one index column found")
    else:
        assert is_index.sum()==1
        id_column = df.columns[is_index][0]
        sample_ids = list(df.loc[data_rows,id_column])
        #fieldinfo_ids = list(df.loc[fieldinfo_rows,id_column])

    # Extract data, field info and sample info
    print("Extracting data")

    data = df.loc[data_rows, data_columns]

    print("Extracting field info")

    field_info = pd.DataFrame(index=data_columns)
    field_info['FieldType'] = columntypes[list(is_data)]
    #for (fieldinfo_name, row_index) in zip(fieldinfo_ids, fieldinfo_rows):
    #    field_info[fieldinfo_name] = df.loc[row_index, data_columns].transpose()

    print("Extracting sample info")

    sample_info = pd.DataFrame(index=data_rows)
    sample_info = df.loc[data_rows, list(is_metadata)]

    # Store datatypes types of sampleinfo fields
    sample_info_types = pd.DataFrame(columntypes[list(is_metadata)])
    assert sample_info_types.shape[1]==1
    sample_info_types.columns = ['InfoType']

    print("Setting data column types")

    # Give numeric columns dtype of float
    # For speed, convert all numeric columns, glue back together,
    # then reorder to original order to match field_info and to match
    # user-specified input order
    is_numeric = field_info['FieldType']=='Q'
    data = pd.merge(data.loc[:,~is_numeric], data.loc[:,is_numeric].astype('float'),
                   left_index=True, right_index=True)[list(field_info.index)]

    # Give categorical columns appropriate dtype (pandas categorical)
    categorical_fields = field_info.index[field_info['FieldType'].isin(['N'])]
    for field in categorical_fields:
        data[field] = data[field].astype('category')

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

    # Use more explicit type specifiers
    field_info['FieldType'] = field_info['FieldType'].map({'Q':'Numeric','N':'Categorical'})
    sample_info_types['InfoType'] = sample_info_types['InfoType'].map({'Q':'Numeric','N':'Categorical'})

    #print("Parse results:")
    #print("Data\n",data)
    #print("sample info",sample_info)
    #print("sample info types",sample_info_types)
    #print("field info",field_info)

    return (data, sample_info, sample_info_types, field_info)

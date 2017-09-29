
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def fill_in_missing(data, field_info, sample_info):
    """
    Fill in missing values, or delete rows/columns, to produce
    a dataset with no missing values.
    """
    # For now test simple methods: just fill in with zeroes and 'Unknown's
    fields_with_missing = data.columns[data.isnull().sum() > 0]

    for field in fields_with_missing:
        print("Filling in missing values in "+field)
        if field_info.loc[field,'FieldType']=='Numeric':
            missing_values = data[field].isnull()
            data.loc[missing_values,field] = 0
        elif field_info.loc[field,'FieldType']=='Categorical':
            missing_values = data[field].isnull()
            new_value = 'Unknown'
            while new_value in data[field].unique():
                new_value = new_value + "_"
            #new_values = ["Unknown{}".format(n+1) for n in range(missing_values.sum())]
            data.loc[missing_values,field] = new_values
    return data

def one_hot(series, categories=None):
    """
    Given a series of M categorical values,
    with N categories,
    return a binary-encoded MxN DataFrame of 0's and 1's,
    where each column corresponds to a category.
    The category name is encoded in the columns of the returned DataFrame,
    i.e. each column name is of form {OriginalFieldName}_{CategoryName}.
    """
    if categories is None:
        vec = series.astype('category')
    else:
        vec = series.astype('category', categories=categories)
    vec_numeric = vec.cat.codes
    encoded = pd.DataFrame(np.eye(len(vec.cat.categories), dtype=int)[vec_numeric])
    encoded.columns = ['{}_{}'.format(series.name, c) for c in vec.cat.categories]
    encoded.index = vec.index
    return encoded

def pca_transform(data, field_info, max_pcs, scale=False):
    """
    Apply PCA to the data. There must be no missing values.
    Return the pca object and the transformed data.
    """
    numeric_fieldspec = field_info['FieldType']=='Numeric'
    categorical_fields = data.columns[field_info['FieldType']=='Categorical']

    if scale:
        # Subtracting mean should have no effect,
        # dividing by std should
        data.loc[:,numeric_fieldspec] -= data.loc[:,numeric_fieldspec].mean()
        data.loc[:,numeric_fieldspec] /= data.loc[:,numeric_fieldspec].std()

    # Encode any categorical fields, and concat results with numerical fields
    # For now, handling only unordered categories
    encoded = pd.concat([data.loc[:,numeric_fieldspec]] +
                        [one_hot(data[field]) for field in categorical_fields],
                        axis=1)
    print("One-hot encoded data shape {}".format(encoded.shape))
    assert np.all(data.index==encoded.index)

    # Do PCA
    num_pcs = min(max_pcs, encoded.shape[1], encoded.shape[0])
    pca = PCA(num_pcs)
    transformed = pd.DataFrame(pca.fit_transform(encoded.as_matrix()), index=encoded.index)
    pca_names = ["PCA{}".format(n) for n in range(1,num_pcs+1)]
    transformed.columns = pca_names
    return (pca, transformed)

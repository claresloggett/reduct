
import pandas as pd
from sklearn.decomposition import PCA

def pca_transform(data, field_info, max_pcs):
    """
    Apply PCA to the data.
    Return the pca object and the transformed data.
    """
    num_pcs = min(max_pcs, data.shape[1])
    pca = PCA(num_pcs)
    transformed = pd.DataFrame(pca.fit_transform(data.as_matrix()), index=data.index)
    pca_names = ["PCA{}".format(n) for n in range(1,num_pcs+1)]
    transformed.columns = pca_names
    return (pca, transformed)

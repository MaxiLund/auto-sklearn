from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
import numpy as np

class OrdinalEncoding(AutoSklearnPreprocessingAlgorithm):
    """
    Substitute missing values by 2
    """

    def __init__(self, random_state=None):
        self.random_stated = random_state

    def fit(self, X, y=None):
        import sklearn.preprocessing

        self.preprocessor = sklearn.preprocessing.OrdinalEncoder()
        self.preprocessor.fit(X)
        categories = self.preprocessor.categories_
        new_categories = []
        for c in categories:
            try:
                # if c is a list of ints, do nothing
                c = c.astype(int)
            except:
                # if c is a list of non-ints, enter category 'rare' as the first element
                c = np.concatenate((['rare'],c))
            new_categories.append(c)

        self.preprocessor.categories_ = new_categories

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        try:

            for col in range(X.shape[1]):
                cat = self.preprocessor.categories_[col]
                try:
                    X[:, col].astype(int)
                    is_ordinal = True
                except Exception:
                    is_ordinal = False

                not_encoded_values = list(False if x in cat else True for x in X[:, col])
                not_encoded_index = np.argwhere(not_encoded_values)
                if is_ordinal:
                    # If the value can be converted to an integer, we look for the closest class based on numerical distance
                    for ix in not_encoded_index:
                        X[ix, col] = self.preprocessor.categories_[col][np.abs(self.preprocessor.categories_[col]-X[ix, col]).argmin()]
                else:
                    # If the value can't be converted to int (
                    for ix in not_encoded_index:
                        X[ix, col] = 'rare'

            X = self.preprocessor.transform(X).astype(int)
        except Exception:
            print(Exception)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'OrdinalEncoder',
                'name': 'Ordinal Encoder',
                'handles_missing_values': False,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()

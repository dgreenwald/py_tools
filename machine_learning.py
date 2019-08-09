from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

class Model:

    def __init__(self, model):

        self.model = model
        
    def set_data(self, mldata):

        self.mldata = mldata

    def fit(self, data=None):

        if data is not None:
            self.set_data(data)
        self.model.fit(self.mldata.train_features, self.mldata.train_labels)

    def predict(self, test_features=None, test_labels=None, display=False):

        if test_features is None:
            test_features = self.mldata.test_features
            test_labels = self.mldata.test_labels

        predictions = self.model.predict(test_features)
        if test_labels is not None:
            errors = np.abs(predictions - test_labels)
            if display: print("Error rate: {:g}".format(np.mean(errors)))
        else:
            errors = None

        return predictions, errors

class Data:

    def __init__(self, train_features, test_features, train_labels, test_labels):

        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels

def encode_dummies(df, old_feature_vars, categorical_vars):

    feature_vars = [var for var in old_feature_vars if var not in categorical_vars]

    for var in categorical_vars:
        this_series = var + '_' + df[var].astype(str)
        dummies = pd.get_dummies(this_series)
        feature_vars += list(dummies.columns)
        df = df.join(dummies)

    return df, feature_vars

def get_labels_features(df, label_var, feature_vars, categorical_vars=[], **kwargs):

    df, feature_vars = encode_dummies(df, feature_vars, categorical_vars)
    train_features, test_features, train_labels, test_labels = train_test_split(
        df[feature_vars].values, df[label_var].values, **kwargs
    )

    return Data(train_features, test_features, train_labels, test_labels)

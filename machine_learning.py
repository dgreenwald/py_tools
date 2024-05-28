# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import KKNImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from utilities import tic, toc
from . import in_out

class RandomForestWrapper:
    
    def __init__(self, data=None, infile=None, rf=None,
                 **kwargs):
        
        self.data = data
        
        if rf is not None:
            self.rf = rf
        elif infile is not None:
            self.load(infile)
        else:
            self.rf = RandomForestClassifier(**kwargs)
            
        # if label_var is not None:
        #     self.set_labels_features(label_var, continuous_vars, category_vars)
            
    def set_data(self, data):
        self.data = data
        
    def set_labels_features(self, label_var, continuous_vars=None,
                            category_vars=None, features_only=False):

        if continuous_vars is None: continuous_vars = []
        if category_vars is None: category_vars = []
        
        self.label_var = label_var
        self.continuous_vars = continuous_vars
        self.category_vars = category_vars
        
        self.labels, self.features, self.names = get_labels_features(
            self.data, label_var, continuous_vars, category_vars, features_only=features_only,
            )
            
    def save(self, outfile):
        in_out.save_pickle(self.rf, outfile)
        
    def load(self, infile):
        self.rf = in_out.load_pickle(infile)
        
    def train_test_split(self, train_size=0.25, test_size=0.75, random_state=17):
        
        if train_size is None:
            
            assert test_size is None
            self.train_features = self.features
            self.train_labels = self.labels
            
            self.test_features = None
            self.test_labels = None
            
        else:
            
            self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
                self.features, self.labels, train_size=train_size, test_size=test_size, random_state=17
            )
        
    def fit(self):
            
        assert self.features is not None
        
        self.rf.fit(self.train_features, self.train_labels)
            
    def predict(self, features=None):
        
        if features is None:
            features = self.features
            
        self.predictions = self.rf.predict(features)
            
    def evaluate(self, test_features=None, test_labels=None, display=True):
        
        if test_features is None:
            test_features = self.test_features
            test_labels = self.test_labels
            
        self.predict(test_features)
        
        # Overall error rate
        errors = np.abs(self.predictions - test_labels)
        self.err_rate = np.mean(errors)
        if display:
            print("Error rate = {:g}".format(self.err_rate))
        
        # Error rate by class
        unique_labels = np.unique(test_labels)
        self.err_rate_by_class = np.zeros(len(unique_labels))
        
        for ii, val in enumerate(unique_labels):
        
            ix = test_labels == val
            self.err_rate_by_class[ii] = np.mean(errors[ix])
            
            if display:
                print("Error rate ({0} = {1}) = {2:g}".format(self.label_var, repr(val), self.err_rate_by_class[ii]))
        
    def plot(self, plotpath=None):
        
        plot_importance_random_forest(self.rf, self.names, plotpath=plotpath)
        
def complete_estimation(df, label_var, continuous_vars=None, category_vars=None, 
                        train_size=0.25, test_size=0.75, outfile=None,
                        evaluate=False, plot=False, plotpath=None, **kwargs):

    if continuous_vars is None: continuous_vars = []
    if category_vars is None: category_vars = []
    
    rfw = RandomForestWrapper(data=df, **kwargs)
    rfw.set_labels_features(label_var, continuous_vars, category_vars)
    rfw.train_test_split(train_size, test_size)
    rfw.fit()
    
    if outfile is not None:
        rfw.save(outfile)
        
    if evaluate:
        rfw.evaluate()
        
    if plot:
        rfw.plot(plotpath)
        
    return rfw

def evaluate_random_forest(rf, test_features, test_labels):

    predictions = rf.predict(test_features)
    errors = np.abs(predictions - test_labels)
    ix_pos = test_labels == 1
    false_pos_rate = np.mean(errors[~ix_pos])
    false_neg_rate = np.mean(errors[ix_pos])

    print("Error rate: {:g}".format(np.mean(errors)))
    print("False positive rate: {:g}".format(false_pos_rate))
    print("False negative rate: {:g}".format(false_neg_rate))
                            
    return predictions 

def estimate_random_forest(rf, labels, features, train_size=0.25, test_size=0.75):
    
    # if rf is None:
    #     rf = RandomForestClassifier(**kwargs)
    
    # if labels is None:
    #     assert features is None
    #     assert df is not None
    #     labels, features = get_labels_features(df, label_var, continuous_vars, category_vars)
    
    # label_vals = labels.values.ravel()
    # feature_vals = features.values
    
    if train_size is None:
        
        assert test_size is None
        train_features = features
        train_labels = labels
        
        test_features = None
        test_labels = None
        
    else:
        
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, train_size=train_size, test_size=test_size, random_state=17
        )

    # start = tic()
    rf.fit(train_features, train_labels)
    # toc(start)
    
    # if evaluate and (test_features is not None):
    #     _ = evaluate_random_forest(rf, test_features, test_labels)
        
    # if plot:
    #     plot_importance_random_forest(rf, features.columns)
    
    return rf

def plot_importance_random_forest(rf, names, plotpath=None):
    
    importances = rf.feature_importances_
    err = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
    indices = np.argsort(importances)[::-1]
    sorted_names = [names[ii] for ii in indices]

    fig = plt.figure()
    plt.bar(sorted_names, importances[indices], alpha=0.5, yerr=err[indices], align='center',
            error_kw={'linewidth' : 2})
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    
    if plotpath is None:
        plt.show()
    else:
        plt.savefig(plotpath)

    plt.close(fig)
    
def get_labels_features(df, label_var, continuous_vars=None,
                        category_vars=None, features_only=False):
    
    if continuous_vars is None: continuous_vars = []
    if category_vars is None: category_vars = []

    feature_list = [df[continuous_vars]]
    for var in category_vars:
        dummies = pd.get_dummies(df[var])
        dummies.columns = [var + '_' + str(ii) for ii in dummies.columns]
        feature_list.append(dummies)
        
    feature_data = pd.concat(feature_list, axis=1)

    if features_only:
        label_vals = None
    else:
        label_vals = df[label_var].values.ravel()
    feature_vals = feature_data.values
    names = list(feature_data.columns)
    
    return label_vals, feature_vals, names

# class Model:

#     def __init__(self, model):

#         self.model = model
        
#     def set_data(self, mldata):

#         self.mldata = mldata

#     def fit(self, data=None):

#         if data is not None:
#             self.set_data(data)
#         self.model.fit(self.mldata.train_features, self.mldata.train_labels)

#     def predict(self, test_features=None, test_labels=None, display=False):

#         if test_features is None:
#             test_features = self.mldata.test_features
#             test_labels = self.mldata.test_labels

#         predictions = self.model.predict(test_features)
#         if test_labels is not None:
#             errors = np.abs(predictions - test_labels)
#             if display: print("Error rate: {:g}".format(np.mean(errors)))
#         else:
#             errors = None

#         return predictions, errors

# class Data:

#     def __init__(self, train_features, test_features, train_labels, test_labels):

#         self.train_features = train_features
#         self.test_features = test_features
#         self.train_labels = train_labels
#         self.test_labels = test_labels

# def encode_dummies(df, old_feature_vars, categorical_vars):

#     feature_vars = [var for var in old_feature_vars if var not in categorical_vars]

#     for var in categorical_vars:
#         this_series = var + '_' + df[var].astype(str)
#         dummies = pd.get_dummies(this_series)
#         feature_vars += list(dummies.columns)
#         df = df.join(dummies)

#     return df, feature_vars

# def get_labels_features(df, label_var, feature_vars, categorical_vars=None, **kwargs):

#     df, feature_vars = encode_dummies(df, feature_vars, categorical_vars)
#     train_features, test_features, train_labels, test_labels = train_test_split(
#         df[feature_vars].values, df[label_var].values, **kwargs
#     )

#     return Data(train_features, test_features, train_labels, test_labels)

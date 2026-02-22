from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from py_tools import in_out

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
                self.features, self.labels, train_size=train_size, test_size=test_size, random_state=random_state
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

    rf.fit(train_features, train_labels)

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

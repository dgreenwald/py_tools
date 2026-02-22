from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from py_tools import in_out

class RandomForestWrapper:
    """
    Wrapper around scikit-learn ``RandomForestClassifier`` with convenience
    methods for data management, training, evaluation, and plotting.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Training/test data.
    infile : str, optional
        Path to a pickled ``RandomForestClassifier`` to load.
    rf : RandomForestClassifier, optional
        Pre-existing classifier instance.  Takes precedence over ``infile``.
    **kwargs
        Additional keyword arguments forwarded to ``RandomForestClassifier``
        when neither ``rf`` nor ``infile`` is supplied.

    Attributes
    ----------
    data : pandas.DataFrame or None
    rf : RandomForestClassifier
    labels : ndarray or None
        Full label array set by :meth:`set_labels_features`.
    features : ndarray or None
        Full feature matrix set by :meth:`set_labels_features`.
    names : list of str or None
        Feature names set by :meth:`set_labels_features`.
    train_features, test_features : ndarray or None
        Split set by :meth:`train_test_split`.
    train_labels, test_labels : ndarray or None
        Split set by :meth:`train_test_split`.
    predictions : ndarray or None
        Predictions set by :meth:`predict`.
    err_rate : float or None
        Overall error rate set by :meth:`evaluate`.
    err_rate_by_class : ndarray or None
        Per-class error rates set by :meth:`evaluate`.
    """

    def __init__(self, data=None, infile=None, rf=None,
                 **kwargs):
        """
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Input data.
        infile : str, optional
            Path to pickled classifier to load.
        rf : RandomForestClassifier, optional
            Pre-existing classifier.
        **kwargs
            Forwarded to ``RandomForestClassifier`` when creating a new one.
        """
        self.data = data
        
        if rf is not None:
            self.rf = rf
        elif infile is not None:
            self.load(infile)
        else:
            self.rf = RandomForestClassifier(**kwargs)

    def set_data(self, data):
        """
        Set the data attribute.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data to store on the wrapper.
        """
        self.data = data
        
    def set_labels_features(self, label_var, continuous_vars=None,
                            category_vars=None, features_only=False):
        """
        Extract labels and features from ``self.data`` and store them.

        Delegates to :func:`get_labels_features`.

        Parameters
        ----------
        label_var : str
            Column name of the target/label variable.
        continuous_vars : list of str, optional
            Names of continuous feature columns.
        category_vars : list of str, optional
            Names of categorical feature columns (expanded into dummies).
        features_only : bool, optional
            If ``True``, skip label extraction (``self.labels`` will be
            ``None``).  Default is ``False``.
        """
        if continuous_vars is None:
            continuous_vars = []
        if category_vars is None:
            category_vars = []
        
        self.label_var = label_var
        self.continuous_vars = continuous_vars
        self.category_vars = category_vars
        
        self.labels, self.features, self.names = get_labels_features(
            self.data, label_var, continuous_vars, category_vars, features_only=features_only,
            )
            
    def save(self, outfile):
        """
        Save the trained classifier to a pickle file.

        Parameters
        ----------
        outfile : str
            Destination file path.
        """
        in_out.save_pickle(self.rf, outfile)
        
    def load(self, infile):
        """
        Load a classifier from a pickle file.

        Parameters
        ----------
        infile : str
            Path to the pickled ``RandomForestClassifier``.
        """
        self.rf = in_out.load_pickle(infile)
        
    def train_test_split(self, train_size=0.25, test_size=0.75, random_state=17):
        """
        Split features and labels into training and test sets.

        If ``train_size`` is ``None`` (and ``test_size`` is ``None``), all
        data is used for training and the test attributes are set to ``None``.

        Parameters
        ----------
        train_size : float or None, optional
            Proportion of data to include in the training split.  Default is
            0.25.
        test_size : float or None, optional
            Proportion of data to include in the test split.  Default is 0.75.
        random_state : int, optional
            Random seed forwarded to ``sklearn.model_selection.train_test_split``.
            Default is 17.
        """
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
        """
        Fit the random forest classifier on the training data.

        Requires :meth:`set_labels_features` and :meth:`train_test_split` to
        have been called first so that ``self.train_features`` and
        ``self.train_labels`` are available.
        """
        assert self.features is not None
        
        self.rf.fit(self.train_features, self.train_labels)
            
    def predict(self, features=None):
        """
        Generate predictions and store them in ``self.predictions``.

        Parameters
        ----------
        features : ndarray, optional
            Feature matrix to predict on.  Defaults to ``self.features``
            (the full dataset).
        """
        if features is None:
            features = self.features
            
        self.predictions = self.rf.predict(features)
            
    def evaluate(self, test_features=None, test_labels=None, display=True):
        """
        Evaluate the classifier on test data and store error rates.

        Parameters
        ----------
        test_features : ndarray, optional
            Feature matrix for evaluation.  Defaults to
            ``self.test_features``.
        test_labels : ndarray, optional
            True labels for evaluation.  Defaults to ``self.test_labels``.
        display : bool, optional
            If ``True`` (default), print overall and per-class error rates.
        """
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
        """
        Plot feature importances for the fitted classifier.

        Parameters
        ----------
        plotpath : str, optional
            File path to save the figure.  If ``None`` (default), the figure
            is displayed interactively.
        """
        plot_importance_random_forest(self.rf, self.names, plotpath=plotpath)
        
def complete_estimation(df, label_var, continuous_vars=None, category_vars=None, 
                        train_size=0.25, test_size=0.75, outfile=None,
                        evaluate=False, plot=False, plotpath=None, **kwargs):
    """
    Full random-forest estimation pipeline.

    Creates a :class:`RandomForestWrapper`, splits the data, fits the model,
    and optionally saves, evaluates, and plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    label_var : str
        Column name of the target variable.
    continuous_vars : list of str, optional
        Names of continuous feature columns.
    category_vars : list of str, optional
        Names of categorical feature columns.
    train_size : float, optional
        Training-set proportion.  Default is 0.25.
    test_size : float, optional
        Test-set proportion.  Default is 0.75.
    outfile : str, optional
        If provided, save the fitted classifier to this pickle path.
    evaluate : bool, optional
        If ``True``, call :meth:`RandomForestWrapper.evaluate` after fitting.
        Default is ``False``.
    plot : bool, optional
        If ``True``, call :meth:`RandomForestWrapper.plot` after fitting.
        Default is ``False``.
    plotpath : str, optional
        File path forwarded to :meth:`RandomForestWrapper.plot`.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`RandomForestWrapper` (and hence to
        ``RandomForestClassifier``).

    Returns
    -------
    rfw : RandomForestWrapper
        Fitted wrapper instance.
    """
    if continuous_vars is None:
        continuous_vars = []
    if category_vars is None:
        category_vars = []
    
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
    """
    Evaluate a trained random forest on test data.

    Prints overall, false-positive, and false-negative error rates and
    returns the predictions array.

    Parameters
    ----------
    rf : RandomForestClassifier
        Fitted classifier.
    test_features : ndarray of shape (n_samples, n_features)
        Feature matrix for the test set.
    test_labels : ndarray of shape (n_samples,)
        True labels for the test set (binary: 0/1).

    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        Class predictions for each test sample.
    """
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
    """
    Train a random forest classifier, optionally with a train/test split.

    Parameters
    ----------
    rf : RandomForestClassifier
        Unfitted (or pre-configured) classifier instance.
    labels : array_like of shape (n_samples,)
        Target labels.
    features : array_like of shape (n_samples, n_features)
        Feature matrix.
    train_size : float or None, optional
        Proportion of data to use for training.  If ``None`` (with
        ``test_size=None``), all data is used for training.  Default is 0.25.
    test_size : float or None, optional
        Proportion of data to use for testing.  Default is 0.75.

    Returns
    -------
    rf : RandomForestClassifier
        The fitted classifier.
    """
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
    """
    Plot feature importances with error bars.

    Displays a bar chart of mean feature importances (sorted descending) with
    standard deviations across trees as error bars.

    Parameters
    ----------
    rf : RandomForestClassifier
        Fitted classifier with ``feature_importances_`` and ``estimators_``
        attributes.
    names : list of str
        Feature names corresponding to columns of the training feature matrix.
    plotpath : str, optional
        File path to save the figure.  If ``None`` (default), the figure is
        displayed interactively via ``plt.show()``.
    """
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
    """
    Extract label array, feature matrix, and feature names from a DataFrame.

    Categorical variables are one-hot encoded via ``pandas.get_dummies``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    label_var : str
        Column name of the target variable.
    continuous_vars : list of str, optional
        Names of continuous feature columns.
    category_vars : list of str, optional
        Names of categorical feature columns to expand as dummies.
    features_only : bool, optional
        If ``True``, skip label extraction and return ``None`` for labels.
        Default is ``False``.

    Returns
    -------
    label_vals : ndarray of shape (n_samples,) or None
        Flattened label array, or ``None`` when ``features_only=True``.
    feature_vals : ndarray of shape (n_samples, n_features)
        Feature matrix.
    names : list of str
        Feature column names (continuous names followed by dummy names).
    """
    if continuous_vars is None:
        continuous_vars = []
    if category_vars is None:
        category_vars = []

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

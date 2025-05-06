#%%
import pandas as pd
import numpy as np
from patsy.highlevel import dmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from statsmodels.stats.outliers_influence import variance_inflation_factor


class SplineLogisticFeatureTransformer:
    def __init__(self, knots=None, lower_bound=None, upper_bound=None, degree=3, include_intercept=False):
        """
        knots: list of knots, if None, will use default quantiles
        degree: degree of spline (default 3 for cubic)
        include_intercept: whether to include intercept in spline basis
        """
        self.knots = knots
        self.degree = degree
        self.include_intercept = include_intercept
        self.basis_formula = "bs(x, degree={}, knots=({},), include_intercept={}, lower_bound={}, upper_bound={})".format(
            self.degree, ",".join(map(str, knots)), str(self.include_intercept), lower_bound, upper_bound
        )
        self.model = None
        self.coef_ = None

    def fit(self, x, y):
        """
        x: 1D array-like feature
        y: 1D array-like label (0/1)
        """
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y)

        # Build spline basis
        spline_basis = dmatrix(self.basis_formula, {"x": x[:, 0]}, return_type="dataframe")

        # Fit logistic regression
        model = LogisticRegression(fit_intercept=True, penalty='l2', solver='lbfgs', C=1.0)
        model.fit(spline_basis, y)

        self.model = model
        # Save coefficients excluding intercept
        self.coef_ = model.coef_.flatten()

        return

    def transform(self, x_new):
        """
        Transform new feature x_new using learned coefficients
        """
        x_new = np.asarray(x_new).reshape(-1, 1)
        spline_basis_new = dmatrix(self.basis_formula, {"x": x_new[:, 0]}, return_type="dataframe")

        # Calculate transformed feature (include model intercept)
        transformed = expit(self.model.intercept_[0] + np.dot(spline_basis_new.values, self.coef_))

        return transformed


def cv_auc_compute(x, y, n_splits=5, random_state=0):
    """
    Perform CV for a full model and return AUCs across folds.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []

    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)

    return np.mean(aucs)


def recursive_vif_elimination(x, threshold=10.0, min_features=5):
    dropped = []

    while True:
        vif = pd.Series(
            [variance_inflation_factor(x.values, i) for i in range(x.shape[1])],
            index=x.columns
        )
        max_vif = vif.max()
        if max_vif <= threshold and len(vif) <= 5:
            break
        drop_feature = vif.idxmax()
        if x.shape[1] < min_features:
            break
        x = x.drop(columns=drop_feature)
        dropped.append(drop_feature)

    return x, dropped, vif


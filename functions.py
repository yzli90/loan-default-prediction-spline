#%%
import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
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


# class LinearScorer(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, 1)  # f(x) = w^T x + b
#
#     def forward(self, x):
#         return self.linear(x).squeeze(1)  # shape: (n,)
#
#
# def pairwise_hinge_loss(scores, labels):
#     pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
#     neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
#     if len(pos_idx) == 0 or len(neg_idx) == 0:
#         return torch.tensor(0.0, requires_grad=True)
#     pos_scores = scores[pos_idx]
#     neg_scores = scores[neg_idx]
#     diffs = - (pos_scores[:, None] - neg_scores[None, :])  # shape: (n_pos, n_neg)
#     losses = torch.clamp(diffs, min=0)
#     return losses.mean()
#
#
# def pairwise_logistic_loss(scores, labels):
#     pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
#     neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
#     if len(pos_idx) == 0 or len(neg_idx) == 0:
#         return torch.tensor(0.0, requires_grad=True)
#     pos_scores = scores[pos_idx]
#     neg_scores = scores[neg_idx]
#     diffs = pos_scores[:, None] - neg_scores[None, :]
#     losses = torch.log1p(torch.exp(-diffs))
#     return losses.mean()
#
#
# def train_pairwise_loss(X_train, y_train, epochs=1000, lr=1e-3, early_stop_rounds=10, tol=1e-5):
#     X_train, y_train = X_train.values, y_train.values
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
#
#     model = LinearScorer(input_dim=X_train.shape[1])
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     best_loss = float('inf')
#     no_improve_rounds = 0
#
#     for epoch in range(epochs):
#         model.train()
#         scores = model(X_train_tensor)
#         loss = pairwise_logistic_loss(scores, y_train_tensor)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
#
#         if best_loss - loss.item() > tol:
#             best_loss = loss.item()
#             no_improve_rounds = 0
#         else:
#             no_improve_rounds += 1
#             if no_improve_rounds >= early_stop_rounds:
#                 print(f"Early stopping at epoch {epoch+1} | Best Loss: {best_loss:.6f}")
#                 break
#
#     return model
#
#
# def predict(model, X):
#     X = X.values
#     model.eval()
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     with torch.no_grad():
#         return model(X_tensor).numpy()






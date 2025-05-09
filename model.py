#%%
import time
from functions import *
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier


years = list(range(2007, 2018+1))
logit_model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='auc', n_jobs=-1, reg_lambda=1.0)
auc = pd.DataFrame(
    index=['Logistic (train)', 'Logistic (test)', 'Logistic_spline (train)', 'Logistic_spline (test)',
           'XGboost (train)', 'XGboost (test)', 'XGboost_spline (train)', 'XGboost_spline (test)'],
    # ,
    # , 'AUC_max (train)', 'AUC_max (test)', 'AUC_max_spline (train)', 'AUC_max_spline (test)'
    columns=years, dtype=float
)
accuracy_rate = pd.DataFrame(
    index=['Logistic (train)', 'Logistic (test)', 'Logistic_spline (train)', 'Logistic_spline (test)',
           'XGboost (train)', 'XGboost (test)', 'XGboost_spline (train)', 'XGboost_spline (test)'],
    # ,
    # , 'AUC_max (train)', 'AUC_max (test)', 'AUC_max_spline (train)', 'AUC_max_spline (test)'
    columns=years, dtype=float
)

for yr0, yr1 in zip(years[:-1], years[1:]):
    print((yr0, yr1))
    start_time = time.time()
    df_train = pd.read_csv(f"Processed data/selected_train_{yr0}.csv", index_col=None)
    df_spline_train = pd.read_csv(f"Processed data/selected(transformed)_train_{yr0}.csv", index_col=None)

    df_test = pd.read_csv(f"Processed data/selected_test_{yr1}.csv", index_col=None)
    df_spline_test = pd.read_csv(f"Processed data/selected(transformed)_test_{yr1}.csv", index_col=None)

    x_train, x_spline_train = df_train.drop(columns=['LoanStatus']), df_spline_train.drop(columns=['LoanStatus'])
    y_train, y_spline_train = df_train['LoanStatus'], df_spline_train['LoanStatus']

    x_test, x_spline_test = df_test.drop(columns=['LoanStatus']), df_spline_test.drop(columns=['LoanStatus'])
    y_test, y_spline_test = df_test['LoanStatus'], df_spline_test['LoanStatus']

    # Logistic Regression
    logit_model.fit(x_train, y_train)
    y_pred_prob_logit = logit_model.predict_proba(x_train)[:, 1]
    y_pred_logit = (y_pred_prob_logit > 0.5).astype(int)
    auc.loc['Logistic (train)', yr0] = roc_auc_score(y_train, y_pred_prob_logit)
    accuracy_rate.loc['Logistic (train)', yr0] = accuracy_score(y_train, y_pred_logit)
    y_pred_prob_logit = logit_model.predict_proba(x_test)[:, 1]
    y_pred_logit = (y_pred_prob_logit > 0.5).astype(int)
    auc.loc['Logistic (test)', yr1] = roc_auc_score(y_test, y_pred_prob_logit)
    accuracy_rate.loc['Logistic (test)', yr1] = accuracy_score(y_test, y_pred_logit)

    logit_model.fit(x_spline_train, y_spline_train)
    y_pred_prob_logit = logit_model.predict_proba(x_spline_train)[:, 1]
    y_pred_logit = (y_pred_prob_logit > 0.5).astype(int)
    auc.loc['Logistic_spline (train)', yr0] = roc_auc_score(y_spline_train, y_pred_prob_logit)
    accuracy_rate.loc['Logistic_spline (train)', yr0] = accuracy_score(y_spline_train, y_pred_logit)
    y_pred_prob_logit = logit_model.predict_proba(x_spline_test)[:, 1]
    y_pred_logit = (y_pred_prob_logit > 0.5).astype(int)
    auc.loc['Logistic_spline (test)', yr1] = roc_auc_score(y_spline_test, y_pred_prob_logit)
    accuracy_rate.loc['Logistic_spline (test)', yr1] = accuracy_score(y_spline_test, y_pred_logit)

    # # AUC maximization
    # auc_model = train_pairwise_loss(x_train, y_train, epochs=100, lr=1e-3)
    # y_pred_prob_auc = expit(predict(auc_model, x_train))
    # y_pred_auc = (y_pred_prob_auc > 0.5).astype(int)
    # auc.loc['AUC_max (train)', yr0] = roc_auc_score(y_train, y_pred_prob_auc)
    # accuracy_rate.loc['AUC_max (train)', yr0] = accuracy_score(y_train, y_pred_auc)
    # y_pred_prob_auc = expit(predict(auc_model, x_test))
    # y_pred_auc = (y_pred_prob_auc > 0.5).astype(int)
    # auc.loc['AUC_max (test)', yr1] = roc_auc_score(y_test, y_pred_prob_auc)
    # accuracy_rate.loc['AUC_max (test)', yr1] = accuracy_score(y_test, y_pred_auc)
    #
    # auc_model = train_pairwise_loss(x_spline_train, y_spline_train, lr=1e-3)
    # y_pred_prob_auc = expit(predict(auc_model, x_spline_train))
    # y_pred_auc = (y_pred_prob_auc > 0.5).astype(int)
    # auc.loc['AUC_max_spline (train)', yr0] = roc_auc_score(y_spline_train, y_pred_prob_auc)
    # accuracy_rate.loc['AUC_max_spline (train)', yr0] = accuracy_score(y_spline_train, y_pred_auc)
    # y_pred_prob_auc = expit(predict(auc_model, x_spline_test))
    # y_pred_auc = (y_pred_prob_auc > 0.5).astype(int)
    # auc.loc['AUC_max_spline (test)', yr1] = roc_auc_score(y_spline_test, y_pred_prob_auc)
    # accuracy_rate.loc['AUC_max_spline (test)', yr1] = accuracy_score(y_spline_test, y_pred_auc)

    # XGBoost
    xgb_model.fit(x_train, y_train)
    y_pred_prob_xgb = xgb_model.predict_proba(x_train)[:, 1]
    y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)
    auc.loc['XGboost (train)', yr0] = roc_auc_score(y_train, y_pred_prob_xgb)
    accuracy_rate.loc['XGboost (train)', yr0] = accuracy_score(y_train, y_pred_xgb)
    y_pred_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]
    y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)
    auc.loc['XGboost (test)', yr1] = roc_auc_score(y_test, y_pred_prob_xgb)
    accuracy_rate.loc['XGboost (test)', yr1] = accuracy_score(y_test, y_pred_xgb)

    xgb_model.fit(x_spline_train, y_spline_train)
    y_pred_prob_xgb = xgb_model.predict_proba(x_spline_train)[:, 1]
    y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)
    auc.loc['XGboost_spline (train)', yr0] = roc_auc_score(y_spline_train, y_pred_prob_xgb)
    accuracy_rate.loc['XGboost_spline (train)', yr0] = accuracy_score(y_spline_train, y_pred_xgb)
    y_pred_prob_xgb = xgb_model.predict_proba(x_spline_test)[:, 1]
    y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)
    auc.loc['XGboost_spline (test)', yr1] = roc_auc_score(y_spline_test, y_pred_prob_xgb)
    accuracy_rate.loc['XGboost_spline (test)', yr1] = accuracy_score(y_spline_test, y_pred_xgb)

    elapsed_time = time.time() - start_time
    print(f"Finished in {elapsed_time:.2f} seconds.")

auc.to_csv('Results/auc.csv', header=True, index=True)
accuracy_rate.to_csv('Results/accuracy_rate.csv', header=True, index=True)




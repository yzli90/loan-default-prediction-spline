#%%
import time
from functions import *
from sklearn.preprocessing import StandardScaler

categoricals = [
    'term', 'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT', 'home_ownership_OTHER',
    'home_ownership_ANY', 'home_ownership_NONE',
    'verification_status_Not Verified',  'purpose_car', 'purpose_credit_card',
    'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house',
    'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_small_business',
    'purpose_vacation', 'purpose_wedding', 'initial_list_status_f', 'application_type_Individual',
    'disbursement_method_Cash', 'verification_status_Verified'
]
group_dict = {
    "home_ownership": [
        'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT', 'home_ownership_OTHER', 'home_ownership_ANY', 'home_ownership_NONE'
    ],
    "purpose": [
        'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
        'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
        'purpose_other', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding'
    ]
}
analytics = ['TotalGain', 'PvGain', 'LifeOfLoan']
years = list(range(2007, 2018+1))
for yr0, yr1 in zip(years[:-1], years[1:]):
    print((yr0, yr1))
    start_time = time.time()
    # --- Load train set ---
    if yr0 == years[0]:
        df_train = pd.read_csv(f"Processed data/processed_{yr0}.csv", index_col=None)
    else:
        df_train = df_test.copy()

    # --- Load test set ---
    df_test = pd.read_csv(f"Processed data/processed_{yr1}.csv", index_col=None)

    features_remove = set(df_train.columns[df_train.nunique(dropna=False) <= 1]) | \
                      set(df_test.columns[df_test.nunique(dropna=False) <= 1])
    df_train = df_train.drop(columns=features_remove & set(df_train.columns))
    df_splines_train = df_train.copy()
    df_test = df_test.drop(columns=features_remove & set(df_test.columns))
    df_splines_test = df_test.copy()

    all_features = list(set(df_train.columns) & set(df_test.columns) - set(analytics))
    features_cont = list(set(all_features) - set(categoricals))
    features_discrete = list(set(all_features) - set(features_cont))
    features_cont.remove('LoanStatus')
    aucs = pd.Series(index=features_cont+list(group_dict.keys()), dtype=float)

    # --- Fit on train, transform on test ---
    for col in features_cont:
        if col not in categoricals:
            x_train = df_train[col].values
            x_test = df_test[col].values

            transformer = SplineLogisticFeatureTransformer(
                knots=np.percentile(x_train, [20, 40, 60, 80]).tolist(),
                lower_bound=min(x_train.min(), x_test.min()),
                upper_bound=max(x_train.max(), x_test.max())
            )
            transformer.fit(x_train, df_train['LoanStatus'].values)
            df_splines_train[col] = transformer.transform(x_train)
            df_splines_test[col] = transformer.transform(x_test)

            aucs.loc[col] = cv_auc_compute(df_splines_train[col].values, df_train['LoanStatus'].values)
    features_cont_picked = aucs.index[aucs >= 0.55].tolist()

    features_discrete_picked = []
    # Find AUC for dummy variables seperated into groups
    for group in group_dict.keys():
        dummy_features = list(set(group_dict[group]) & set(features_discrete))
        x_train = df_train[dummy_features].values

        aucs.loc[group] = cv_auc_compute(x_train, df_train['LoanStatus'].values)
        if aucs.loc[group] >= 0.55:
            features_discrete_picked += dummy_features[1:]  # drop baseline

    aucs = aucs.sort_values(ascending=False)
    # features_picked = features_cont_picked + features_discrete_picked
    features_picked = features_cont_picked

    # Deal with multicollinearity
    _, features_dropped, vif = recursive_vif_elimination(
        df_train[features_picked], threshold=10, min_features=3
    )
    features_picked = list(set(features_picked) - set(features_dropped))
    features_cont_picked = list(set(features_cont_picked) - set(features_dropped))
    # features_discrete_picked = list(set(features_discrete_picked) - set(features_dropped))
    pd.concat([df_splines_train[['LoanStatus']], df_splines_train[features_picked]], axis=1).to_csv(
        f"Processed data/selected(transformed)_train_{yr0}.csv", header=True, index=False
    )
    pd.concat([df_splines_test[['LoanStatus']], df_splines_test[features_picked]], axis=1).to_csv(
        f"Processed data/selected(transformed)_test_{yr1}.csv", header=True, index=False
    )
    scaler = StandardScaler()
    pd.concat(
        [df_train[['LoanStatus']], pd.DataFrame(
            scaler.fit_transform(df_train[features_cont_picked]), columns=features_cont_picked, index=df_train.index
        )], axis=1
    ).to_csv(f"Processed data/selected_train_{yr0}.csv", header=True, index=False)
    pd.concat(
        [df_test[['LoanStatus']], pd.DataFrame(
            scaler.fit_transform(df_test[features_cont_picked]), columns=features_cont_picked, index=df_test.index
        )], axis=1
    ).to_csv(f"Processed data/selected_test_{yr1}.csv", header=True, index=False)
    elapsed_time = time.time() - start_time
    print(features_picked)
    print(vif)
    print(f"Finished in {elapsed_time:.2f} seconds.")







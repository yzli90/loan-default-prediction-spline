#%%
import pandas as pd
import os
import time
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def null_values(df):
    mis_val = df.isnull().sum()
    mis_val_table = pd.concat([mis_val, 100 * mis_val / len(df)], axis=1)
    mis_val_table.columns = ['Missing Counts', '% of Total Counts']
    mis_val_table = mis_val_table.sort_values('% of Total Counts', ascending=True)
    return mis_val_table


def pv(q, rate):
    discount = 1 / (1 + rate)
    pv = 0
    for value in reversed(q):
        pv *= discount
        pv += value
    return pv


def calc_pv(row):
    if row['LifeOfLoan'] == 0:
        return row['TotalGain']
    eq_pymnt = row['TotalGain'] / row['LifeOfLoan']
    q = [eq_pymnt] * int(row['LifeOfLoan'])
    return pv(q, mon_rate)


for yr in [str(x) for x in range(2007, 2018+1)]:
    print(yr)
    start_time = time.time()
    csv_file = os.path.join("Raw data/loan_{}.csv".format(yr))

    # Load file
    print("Reading csv file {}".format(csv_file))
    df = pd.read_csv(csv_file, low_memory=False)
    print("Shape {}".format(df.shape))

    # Use only those loans that are either Fully Paid or Charged Off
    print(df.loan_status.value_counts())
    fully_paid_df = df[df['loan_status'] == 'Fully Paid']
    charged_off_df = df[df['loan_status'] == 'Charged Off']
    local_df = fully_paid_df.append(charged_off_df)
    print("Total loans either Fully Paid or charged off: {}".format(local_df.shape[0]))
    print("Charged off percent: {0:.2f}%".format(
        100 * charged_off_df.shape[0] / local_df.shape[0])
    )

    # Refactor employment length into numbers (yr) and fill Null values - this might be an important indicator
    local_df['emp_length'].fillna(value=0, inplace=True)
    local_df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    local_df['emp_length'].replace(to_replace='', value=0, inplace=True)
    local_df.loc[:, 'emp_length'] = local_df.loc[:, 'emp_length'].astype(int)

    # Refactor loan term into numbers (yr) and fill Null values
    local_df['term'].fillna(value=0, inplace=True)
    local_df['term'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    local_df['term'].replace(to_replace='', value=0, inplace=True)
    local_df.loc[:, 'term'] = local_df.loc[:, 'term'].astype(int)

    # Remove columns with too many missing values
    percent_missing_threshold = 30
    miss_values = null_values(local_df)
    cols_to_remove = miss_values[miss_values['% of Total Counts'] > percent_missing_threshold].index
    # print("Removing columns: {}".format(cols_to_remove))
    local_df.drop(cols_to_remove, axis=1, inplace=True)

    # remove rows with null entries in dates
    rows = local_df.shape[0]
    local_df = local_df.loc[~local_df.issue_d.isnull()]
    local_df = local_df.loc[~local_df.last_pymnt_d.isnull()]
    local_df = local_df.loc[~local_df.last_pymnt_d.isna()]
    local_df = local_df.loc[~local_df.earliest_cr_line.isnull()]
    local_df = local_df.loc[~local_df.last_credit_pull_d.isnull()]
    print("Removed {0:.2f}% of rows because of null entries in date columns".format(100*(rows - local_df.shape[0])/rows))
    print(local_df.shape)

    # Transform dates to datetime format
    local_df['issue_d'] = pd.to_datetime(local_df.issue_d, format='%b-%Y')
    local_df['last_pymnt_d'] = pd.to_datetime(local_df.last_pymnt_d, format='%b-%Y')
    local_df['earliest_cr_line'] = pd.to_datetime(local_df.earliest_cr_line, format='%b-%Y')
    local_df['last_credit_pull_d'] = pd.to_datetime(local_df.last_credit_pull_d, format='%b-%Y')

    # Number the months from earliest issue_d to latest last_pymnt_d
    local_df['issue_month'] = local_df.issue_d.apply(lambda x: 12*x.year + x.month).astype(int)
    local_df['last_p_month'] = local_df.last_pymnt_d.apply(lambda x: 12*x.year + x.month).astype(int)
    local_df['earliest_cr_line_month'] = local_df.earliest_cr_line.apply(lambda x: 12*x.year + x.month).astype(int)
    local_df['last_credit_pull_month'] = local_df.last_credit_pull_d.apply(lambda x: 12*x.year + x.month).astype(int)
    local_df['earliest_cr_line_age'] = (local_df.issue_month - local_df.earliest_cr_line_month).apply(lambda x: max(x,0))
    issue_months = min(local_df.issue_month)  # Normalize so that issue_month is month 0
    local_df['issue_month'] -= issue_months
    local_df['last_p_month'] -= issue_months
    local_df['earliest_cr_line_month'] -= issue_months
    local_df['last_credit_pull_month'] -= issue_months

    unique_i_d = sorted(pd.to_datetime(local_df.issue_d.unique()))
    unique_p_d = sorted(pd.to_datetime(local_df.last_pymnt_d.unique()))
    dates = set(unique_i_d + unique_p_d)

    # Print out some date stats
    earliest_issue_d = min(unique_i_d)
    latest_issue_d = max(unique_i_d)
    earliest_last_payment_d = min(unique_p_d)
    latest_last_payment_d = max(unique_p_d)
    print("Issue dates from: {} to: {}".format(earliest_issue_d.strftime("%b-%Y"), latest_issue_d.strftime("%b-%Y")))
    print("Last payment dates from: {} to: {}".format(earliest_last_payment_d.strftime("%b-%Y"),
                                                      latest_last_payment_d.strftime("%b-%Y")))

    # Calculate analytical quantities
    analytics = ['TotalGain', 'PvGain', 'LoanStatus', 'LifeOfLoan']
    # Total Gain
    #   principal received minus funded amount
    #   plus interest
    #   plus recoveries in case of default
    #   minus collection recovery fee
    local_df['TotalGain'] = local_df['total_rec_prncp'] - local_df['funded_amnt'] + local_df['total_rec_int'] + \
                            local_df['recoveries'] - local_df['collection_recovery_fee']

    # loan_status:local_df
    #    ChargedOff = 1
    #    Fully Paid = 0
    local_df['LoanStatus'] = (local_df['loan_status'] != 'Fully Paid').astype(int)

    # Life of loan
    #   begins on issue month
    #   ends on last payment month
    #   ... so if loan issued on month 0 and ends on month 1, then there is one payment recieved.
    local_df['LifeOfLoan'] = local_df.last_p_month - local_df.issue_month

    # Present Value - Assume equal monthly payments totalling total gain for present value calculation
    ann_rate = 2.0
    mon_rate = ann_rate / 12
    local_df['PvGain'] = local_df.apply(calc_pv, axis=1)

    # Encode the sub_grade
    grade_mapping = {}
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G']):
        for j in range(1, 6):
            grade_mapping[f"{letter}{j}"] = 5*i + j
    # local_df['sub_grade'] = local_df['sub_grade'].map(grade_mapping).astype('int')
    sub_grades = local_df['sub_grade'].map(grade_mapping).astype('int')

    # print(analytics)
    # print(sorted(local_df.columns))

    # Feature segmentations
    datetimetypes = ['issue_d', 'last_pymnt_d', 'earliest_cr_line', 'last_credit_pull_d']
    # date_identifiers = ['issue_month', 'last_p_month','earliest_cr_line_month', 'last_credit_pull_month',
    #                     'int_rate', 'installment']
    date_identifiers = ['issue_month', 'last_p_month', 'earliest_cr_line_month', 'last_credit_pull_month']
    redundants = ['grade']
    not_possible_to_know_at_decision_time = [
        'funded_amnt', 'funded_amnt_inv',
        'collection_recovery_fee', 'debt_settlement_flag',
        'hardship_flag',
        'last_pymnt_amnt', 'loan_status',
        'out_prncp', 'out_prncp_inv',
        'pymnt_plan', 'recoveries', 'total_pymnt', 'total_pymnt_inv',
        'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp'
    ]
    too_many = ['id', 'emp_title', 'title', 'addr_state', 'zip_code', 'url']
    if 'desk' in local_df.columns:
        too_many.append('desk')

    exclude_features = datetimetypes + date_identifiers + redundants + not_possible_to_know_at_decision_time + too_many
    features = [col for col in local_df.columns if col not in exclude_features]

    local_df[features].select_dtypes(['object']).apply(pd.Series.nunique, axis=0)
    encoded_df = local_df[features].copy()
    exclude_df = local_df[exclude_features].copy()

    # label encode the binary categories
    le = LabelEncoder()
    binary_cols = [col for col in features if encoded_df[col].dtype == 'object' and encoded_df[col].nunique() == 2]
    for col in binary_cols:
        encoded_df[col] = le.fit_transform(encoded_df[col])
    print(f"{len(binary_cols)} binary columns were label encoded.")

    # One-hot encode the rest of the object columns
    encoded_df = pd.get_dummies(encoded_df.drop(columns='sub_grade'), sparse=False)

    # Use SimpleImputer to handle bad entries
    imputer = SimpleImputer(strategy='mean')
    encoded_df = pd.DataFrame(imputer.fit_transform(encoded_df), columns=encoded_df.columns, index=encoded_df.index)

    features_encoded = list(encoded_df.columns)

    # Glue the encoded_df and exclude_df back together
    data_df = pd.concat([encoded_df, sub_grades], axis=1, sort=False)

    # features to delete
    features_to_delete = [
        'acc_now_delinq', 'acc_open_past_24mths', 'application_type', 'avg_cur_bal',
        'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
        'collections_12_mths_ex_med',
        'disbursement_method',
        'delinq_2yrs', 'delinq_amnt',
        'initial_list_status',
        'inq_last_6mths',  'mort_acc',
        'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
        'mths_since_recent_bc', 'mths_since_recent_inq',
        'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
        'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
        'num_accts_ever_120_pd','num_actv_bc_tl', 'num_actv_rev_tl',
        'open_acc',
        'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'policy_code', 'pub_rec',  'pub_rec_bankruptcies',
        'revol_bal', 'revol_util',
        'tax_liens',
        'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc', 'total_bal_ex_mort',
        'total_bc_limit', 'total_il_high_credit_limit',  'total_rev_hi_lim'
    ]
    core_features = [col for col in data_df.columns if col not in features_to_delete + exclude_features]

    data_df[core_features].to_csv(
        os.path.join("Processed data/processed_"+yr+".csv"), header=True, index=False
    )
    elapsed_time = time.time() - start_time
    print(f"Year {yr} finished in {elapsed_time:.2f} seconds.")




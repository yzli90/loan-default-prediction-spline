#%%
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
#%% Compare performance
# Read
auc = pd.read_csv('Results/auc.csv', index_col=0)
accuracy_rate = pd.read_csv('Results/accuracy_rate.csv', index_col=0)

# Figure 1: AUC subplot（In-sample vs Out-of-sample）
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig1.suptitle('Original vs Transformed (AUC)')

axes1[0].set_title('In-sample')
axes1[1].set_title('Out-of-sample')

for idx in auc.index:
    if 'train' in idx:
        axes1[0].plot(auc.columns, auc.loc[idx], label=idx.replace(' (train)', ""))
    else:
        axes1[1].plot(auc.columns, auc.loc[idx], label=idx.replace(' (test)', ""))

for ax in axes1:
    ax.set_ylabel('AUC')
    ax.grid(axis='y')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Results/auc_comparisons.png")

# Figure 2: Accuracy subplot（In-sample vs Out-of-sample）
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig2.suptitle('Original vs Transformed (Accuracy Rate)')

axes2[0].set_title('In-sample')
axes2[1].set_title('Out-of-sample')

for idx in accuracy_rate.index:
    if 'train' in idx:
        axes2[0].plot(accuracy_rate.columns, accuracy_rate.loc[idx], label=idx.replace(' (train)', ""))
    else:
        axes2[1].plot(accuracy_rate.columns, accuracy_rate.loc[idx], label=idx.replace(' (test)', ""))

for ax in axes2:
    ax.set_ylabel('Accuracy Rate')
    ax.grid(axis='y')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Results/accuracy_comparisons.png")
#%% Feature importance
auc_df = pd.read_csv("Results/feature_cv_auc.csv", index_col=0)
auc_df['mean_auc'] = auc_df.mean(axis=1, skipna=True)
top_features = auc_df['mean_auc'].sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel('Mean CV AUC')
plt.title('Top 10 Features by Mean CV AUC')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("Results/feature_importance.png")
#%% Feature selected frequency
df = pd.read_csv("Results/selected_features_by_year.csv", index_col=0)

# Flatten all selected feature names into a list
all_selected = df.values.flatten()
all_selected = [feat for feat in all_selected if pd.notna(feat)]

# Count frequency
feature_counts = Counter(all_selected)
top_counts = pd.Series(feature_counts).sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(8, 6))
top_counts.sort_values().plot(kind='barh', color='darkorange')
plt.xlabel('Number of Years Selected')
plt.title('Top 10 Features by Selection Frequency')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("Results/feature_frequencies.png")
#%% Spline plot
yr = 2015
df = pd.read_csv(f"Processed data/processed_{yr}.csv", index_col=None)
df_spline = pd.read_csv(f"Processed data/selected(transformed)_train_{yr}.csv", index_col=None)

x, x_spline = df.drop(columns=['LoanStatus']), df_spline.drop(columns=['LoanStatus'])
y, y_spline = df['LoanStatus'], df_spline['LoanStatus']

features_picked = list(set(top_counts.index[:5]) & set(x_spline.columns))
for col in features_picked:
    plt.figure()
    x_sorted = x[col].sort_values(ascending=True)
    if x[col].nunique() > 50:
        mask = (x_sorted >= x_sorted.quantile(0.03)) & (x_sorted <= x_sorted.quantile(0.97))
        x_plot = x_sorted[mask]
        plt.plot(x_plot, x_spline.loc[x_plot.index, col], label='Transformed Feature', linewidth=2)
        knot_indices = x_sorted[x_sorted.isin(x_sorted.quantile([0.2, 0.4, 0.6, 0.8]))].index
        plt.scatter(
            x_sorted.loc[knot_indices], x_spline.loc[knot_indices, col], color='red', s=50, label='Knots', zorder=5
        )
    else:
        plt.scatter(x_sorted, x_spline.loc[x_sorted.index, col], label='Transformed Feature', s=10)
    plt.xlabel(col)
    plt.ylabel(col + ' transformed')
    plt.title('Feature Transformation via Spline + Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Results/transformed_feature - {col}.png")









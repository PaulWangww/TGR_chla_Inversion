"""NDCI-Chl-a empirical algorithm construction.

Builds regression models linking NDCI to Chl-a concentration.

Workflow:
  1. Data loading and stratified train/test split (2/3 train, 1/3 test)
  2. Four candidate models + stratified 5-fold CV (3 repeats)
  3. Model selection by best CV-RMSE
  4. Final model fitting and test-set evaluation
  5. Residual diagnostics
  6. Visualization
  7. Save results to timestamped output folder
"""

import os
import sys
import warnings
import datetime
import time
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, shapiro, ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================
# 0. Configuration
# ============================================
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "result", "2_matched", "matched&Gloria.xlsx")
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "result", "3_model", "test")

# Use fixed output folder
OUTPUT_DIR = OUTPUT_BASE

RANDOM_STATE = 42
TEST_SIZE = 1/3  # 1/3 for testing
N_SPLITS = 5     # 5-fold CV
N_REPEATS = 3    # repeat 3 times

indicator_name = 'Chl-a'
indicator_unit = 'ug/L'

# Create subdirectory structure
dirs = {
    'data': os.path.join(OUTPUT_DIR, '1_Data'),
    'cv': os.path.join(OUTPUT_DIR, '2_CV_Results'),
    'model': os.path.join(OUTPUT_DIR, '3_Final_Model'),
    'eval': os.path.join(OUTPUT_DIR, '4_Test_Evaluation'),
    'figs': os.path.join(OUTPUT_DIR, '5_Figures'),
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("NDCI-Chl-a Algorithm Construction")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)


# ============================================
# 1. Data loading and preprocessing
# ============================================
print("\n" + "=" * 60)
print("Step 1: Data loading and preprocessing")
print("=" * 60)

df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
print(f"  Raw data: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing values: {df.isnull().sum().sum()}")

# Basic statistics
stats_desc = df[['chla', 'NDCI']].describe()
print(f"\n  Chl-a statistics:\n{stats_desc['chla']}")
print(f"\n  NDCI statistics:\n{stats_desc['NDCI']}")

# Save data statistics
with open(os.path.join(dirs['data'], 'data_statistics.txt'), 'w', encoding='utf-8') as f:
    f.write("=== Data Statistics ===\n\n")
    f.write(f"Total samples: {df.shape[0]}\n")
    f.write(f"Number of sites: {df['site'].nunique()}\n")
    f.write(f"Sites: {list(df['site'].unique())}\n\n")
    f.write("Chl-a (ug/L):\n")
    f.write(f"  Min: {df['chla'].min():.4f}\n")
    f.write(f"  Max: {df['chla'].max():.4f}\n")
    f.write(f"  Mean: {df['chla'].mean():.4f}\n")
    f.write(f"  Median: {df['chla'].median():.4f}\n")
    f.write(f"  Std: {df['chla'].std():.4f}\n")
    f.write(f"  Skewness: {df['chla'].skew():.4f}\n\n")
    f.write("NDCI:\n")
    f.write(f"  Min: {df['NDCI'].min():.4f}\n")
    f.write(f"  Max: {df['NDCI'].max():.4f}\n")
    f.write(f"  Mean: {df['NDCI'].mean():.4f}\n")
    f.write(f"  Median: {df['NDCI'].median():.4f}\n")
    f.write(f"  Std: {df['NDCI'].std():.4f}\n")
    f.write("  Note: All models use raw NDCI values; no shift constant is needed.\n")

# ============================================
# 2. Stratified train/test split (2/3 vs 1/3)
# ============================================
print("\n" + "=" * 60)
print("Step 2: Stratified train/test split")
print("=" * 60)

# Stratify by Chl-a quantiles
df_sorted = df.sort_values('chla').reset_index(drop=True)
df_sorted['stratum'] = pd.qcut(df_sorted['chla'], q=5, labels=False, duplicates='drop')
n_strata = df_sorted['stratum'].nunique()
print(f"  Number of strata: {n_strata}")

# Sample 2/3 : 1/3 within each stratum
np.random.seed(RANDOM_STATE)
train_indices = []
test_indices = []

for s in range(n_strata):
    stratum_idx = df_sorted[df_sorted['stratum'] == s].index.values
    stratum_size = len(stratum_idx)
    n_train = int(round(stratum_size * (1 - TEST_SIZE)))
    # Ensure at least 1 test sample
    n_test = stratum_size - n_train
    if n_test < 1:
        n_train = stratum_size - 1
        n_test = 1
    perm = np.random.permutation(stratum_idx)
    train_indices.extend(perm[:n_train])
    test_indices.extend(perm[n_train:n_train + n_test])

# Map back to original DataFrame
df_train = df_sorted.loc[train_indices].sort_index()
df_test = df_sorted.loc[test_indices].sort_index()

# Verify distribution consistency
ks_stat, ks_pval = ks_2samp(df_train['chla'], df_test['chla'])
print(f"  Training set: {len(df_train)} samples")
print(f"  Test set: {len(df_test)} samples")
print(f"  KS test (Chl-a): stat={ks_stat:.4f}, p={ks_pval:.4f}")

# Save split results
split_df = pd.DataFrame({
    'dataset': ['train'] * len(df_train) + ['test'] * len(df_test),
    'site': pd.concat([df_train['site'], df_test['site']]),
    'datetime': pd.concat([df_train['datetime'], df_test['datetime']]),
    'chla': pd.concat([df_train['chla'], df_test['chla']]),
    'NDCI': pd.concat([df_train['NDCI'], df_test['NDCI']]),
}).reset_index(drop=True)
split_df.to_csv(os.path.join(dirs['data'], 'train_test_split.csv'), index=False, encoding='utf-8-sig')

# Features and target
X_train = df_train[['NDCI']].values
y_train = df_train['chla'].values
X_test = df_test[['NDCI']].values
y_test = df_test['chla'].values
X_all = np.concatenate([X_train, X_test])
y_all = np.concatenate([y_train, y_test])


# ============================================
# 3. Candidate model definitions
# ============================================
print("\n" + "=" * 60)
print("Step 3: Candidate model definitions (4 models, no shift needed)")
print("=" * 60)


class LogLinearModel(BaseEstimator, RegressorMixin):
    """ln(Chl-a) = b0 + b1*NDCI, with Duan's smearing correction."""
    def __init__(self):
        self.model = LinearRegression()
        self.smearing_factor_ = 1.0

    def fit(self, X, y):
        log_y = np.log(np.maximum(y, 1e-10))
        self.model.fit(X, log_y)
        log_pred = self.model.predict(X)
        residuals = log_y - log_pred
        self.smearing_factor_ = np.mean(np.exp(residuals))
        return self

    def predict(self, X):
        log_pred = self.model.predict(X)
        return np.exp(log_pred) * self.smearing_factor_


# Model registry (only models not requiring a shift constant)
models = {
    'M1_Linear': Pipeline([('reg', LinearRegression())]),
    'M2_Quadratic': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('reg', LinearRegression())
    ]),
    'M3_Cubic': Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('reg', LinearRegression())
    ]),
    'M4_LogLinear': LogLinearModel(),
}

print(f"  {len(models)} candidate models: {list(models.keys())}")


# ============================================
# 4. Stratified repeated K-fold cross-validation
# ============================================
print("\n" + "=" * 60)
print("Step 4: Stratified 5-fold CV (3 repeats)")
print("=" * 60)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    # Avoid issues from negative predictions
    y_pred = np.maximum(y_pred, 0)

    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MRE: Mean Relative Error
    mask = y_true > 0
    if mask.sum() > 0:
        mre = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mre = np.nan

    # Bias
    bias = np.mean(y_pred - y_true)

    # NSE
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'r': r, 'R2': r2, 'RMSE': rmse, 'MAE': mae,
        'MRE': mre, 'Bias': bias, 'NSE': nse
    }


# Build stratified CV folds
def get_stratified_folds(X, y, n_splits=5, random_state=42):
    """Stratified fold assignment based on y."""
    n = len(y)
    # Stratify by quantiles
    y_series = pd.Series(y.flatten() if hasattr(y, 'flatten') else y)
    strata = pd.qcut(y_series, q=n_splits, labels=False, duplicates='drop')
    actual_splits = strata.nunique()

    indices = np.arange(n)
    folds = [[] for _ in range(actual_splits)]

    for s in range(actual_splits):
        s_idx = indices[strata == s]
        np.random.seed(random_state + s)
        np.random.shuffle(s_idx)
        folds[s] = list(s_idx)
 
    # If actual strata < n_splits, fill with simple random splits
    if actual_splits < n_splits:
        # Fill remaining samples
        remaining = [i for i in indices if i not in [item for sublist in folds for item in sublist]]
        for i, idx in enumerate(remaining):
            folds[i % actual_splits].append(idx)
        return folds

    return folds


# Store CV results
cv_results = {name: {'rmse': [], 'mae': [], 'mre': [], 'r': [], 'r2': []}
              for name in models.keys()}
detailed_cv = []

for repeat in range(N_REPEATS):
    seed = RANDOM_STATE + repeat * 10
    folds = get_stratified_folds(X_train, y_train, n_splits=N_SPLITS, random_state=seed)

    for fold_idx, fold_indices in enumerate(folds):
        val_idx = fold_indices
        train_idx = [i for i in range(len(X_train)) if i not in val_idx]

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
            continue

        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                y_pred_val = model.predict(X_val)
                y_pred_val = np.maximum(y_pred_val, 0).flatten()
                metrics = compute_metrics(y_val, y_pred_val)

                cv_results[name]['rmse'].append(metrics['RMSE'])
                cv_results[name]['mae'].append(metrics['MAE'])
                cv_results[name]['mre'].append(metrics['MRE'])
                cv_results[name]['r'].append(metrics['r'])
                cv_results[name]['r2'].append(metrics['R2'])

                detailed_cv.append({
                    'repeat': repeat + 1, 'fold': fold_idx + 1, 'model': name,
                    **metrics
                })
            except Exception as e:
                print(f"    {name} fold {fold_idx+1} failed: {e}")

    print(f"  Repeat {repeat + 1}/{N_REPEATS} completed")

# Summarize CV results
cv_summary = []
for name in models.keys():
    rmse_arr = np.array(cv_results[name]['rmse'])
    mae_arr = np.array(cv_results[name]['mae'])
    mre_arr = np.array(cv_results[name]['mre'])
    r_arr = np.array(cv_results[name]['r'])
    r2_arr = np.array(cv_results[name]['r2'])

    # Remove NaNs
    rmse_arr = rmse_arr[~np.isnan(rmse_arr)]
    mae_arr = mae_arr[~np.isnan(mae_arr)]

    if len(rmse_arr) == 0:
        continue

    cv_summary.append({
        'Model': name,
        'CV_RMSE_mean': np.mean(rmse_arr),
        'CV_RMSE_std': np.std(rmse_arr),
        'CV_MAE_mean': np.mean(mae_arr),
        'CV_MAE_std': np.std(mae_arr),
        'CV_MRE_mean': np.mean(mre_arr[~np.isnan(mre_arr)]) if len(mre_arr[~np.isnan(mre_arr)]) > 0 else np.nan,
        'CV_r_mean': np.mean(r_arr),
        'CV_R2_mean': np.mean(r2_arr),
    })

cv_df = pd.DataFrame(cv_summary).sort_values('CV_RMSE_mean')
print(f"\n  CV summary (sorted by CV-RMSE):")
print(cv_df[['Model', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_r_mean']].to_string(index=False))

# Save CV results
cv_df.to_csv(os.path.join(dirs['cv'], 'cv_results_summary.csv'), index=False, encoding='utf-8-sig')
detailed_cv_df = pd.DataFrame(detailed_cv)
detailed_cv_df.to_csv(os.path.join(dirs['cv'], 'cv_results_detailed.csv'), index=False, encoding='utf-8-sig')

# Model selection: best by CV-RMSE
best_idx = cv_df['CV_RMSE_mean'].idxmin()
selected_model_name = cv_df.loc[best_idx, 'Model']
best_rmse = cv_df.loc[best_idx, 'CV_RMSE_mean']
best_se = cv_df.loc[best_idx, 'CV_RMSE_std']

print(f"\n  Selected model (best CV-RMSE): {selected_model_name} (RMSE={best_rmse:.4f}, SE={best_se:.4f})")

# Save model selection rationale
with open(os.path.join(dirs['cv'], 'model_selection.txt'), 'w', encoding='utf-8') as f:
    f.write("=== Model Selection Report ===\n\n")
    f.write(f"CV method: {N_SPLITS}-fold stratified CV, {N_REPEATS} repeats\n")
    f.write(f"Selection criterion: smallest CV-RMSE\n\n")
    f.write("All models CV-RMSE:\n")
    for _, row in cv_df.iterrows():
        f.write(f"  {row['Model']}: CV-RMSE = {row['CV_RMSE_mean']:.4f} +/- {row['CV_RMSE_std']:.4f}\n")
    f.write(f"\nBest model: {selected_model_name} (CV-RMSE={best_rmse:.4f} +/- {best_se:.4f})\n")


# ============================================
# 5. Final model fitting and test-set evaluation
# ============================================
print("\n" + "=" * 60)
print("Step 5: Final model fitting and test-set evaluation")
print("=" * 60)

final_model = models[selected_model_name]
final_model.fit(X_train, y_train)

# Training-set predictions
y_pred_train = final_model.predict(X_train)
y_pred_train = np.maximum(y_pred_train, 0).flatten()

# Test-set predictions
y_pred_test = final_model.predict(X_test)
y_pred_test = np.maximum(y_pred_test, 0).flatten()

# Compute metrics
train_metrics = compute_metrics(y_train, y_pred_train)
test_metrics = compute_metrics(y_test, y_pred_test)

print(f"\n  Training-set metrics:")
print(f"    r={train_metrics['r']:.4f}, R2={train_metrics['R2']:.4f}")
print(f"    RMSE={train_metrics['RMSE']:.4f}, MAE={train_metrics['MAE']:.4f}")
print(f"    MRE={train_metrics['MRE']:.2f}%, NSE={train_metrics['NSE']:.4f}")

print(f"\n  Test-set metrics:")
print(f"    r={test_metrics['r']:.4f}, R2={test_metrics['R2']:.4f}")
print(f"    RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}")
print(f"    MRE={test_metrics['MRE']:.2f}%, NSE={test_metrics['NSE']:.4f}")

# Save test predictions
test_pred_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred_test.flatten(),
    'Residual': y_test.flatten() - y_pred_test.flatten(),
    'NDCI': X_test.flatten(),
})
test_pred_df.to_csv(os.path.join(dirs['eval'], 'test_predictions.csv'), index=False, encoding='utf-8-sig')

# Save test metrics
with open(os.path.join(dirs['eval'], 'test_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("=== Test Set Evaluation ===\n\n")
    f.write(f"Model: {selected_model_name}\n\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\n\n=== Training Set Metrics ===\n\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")

# Extract model coefficients (if available)
model_eq_path = os.path.join(dirs['model'], 'model_equation.txt')
with open(model_eq_path, 'w', encoding='utf-8') as f:
    f.write("=== Final Model Equation ===\n\n")
    f.write(f"Model: {selected_model_name}\n\n")

    if selected_model_name == 'M1_Linear':
        coef = final_model.named_steps['reg'].coef_
        intercept = final_model.named_steps['reg'].intercept_
        eq = f"Chl-a = {intercept:.6f} + {coef[0]:.6f} * NDCI"
        f.write(f"{eq}\n")
        print(f"\n  Model equation: {eq}")

    elif selected_model_name in ['M2_Quadratic', 'M3_Cubic']:
        coef = final_model.named_steps['reg'].coef_
        intercept = final_model.named_steps['reg'].intercept_
        if selected_model_name == 'M2_Quadratic':
            eq = f"Chl-a = {intercept:.6f} + {coef[0]:.6f}*NDCI + {coef[1]:.6f}*NDCI^2"
        else:
            eq = (f"Chl-a = {intercept:.6f} + {coef[0]:.6f}*NDCI "
                  f"+ {coef[1]:.6f}*NDCI^2 + {coef[2]:.6f}*NDCI^3")
        f.write(f"{eq}\n")
        print(f"\n  Model equation: {eq}")

    elif selected_model_name == 'M4_LogLinear':
        coef = final_model.model.coef_
        intercept = final_model.model.intercept_
        sf = final_model.smearing_factor_
        eq_ln = (f"ln(Chl-a) = {intercept:.6f} + {coef[0]:.6f} * NDCI, "
                 f"smearing={sf:.4f}")
        eq = (f"Chl-a = exp({intercept:.6f} + {coef[0]:.6f}*NDCI) * {sf:.4f}")
        f.write(f"{eq_ln}\n")
        f.write(f"Back-transform: {eq}\n")
        print(f"\n  Model equation: {eq}")

    # Note: All models use raw NDCI values; no shift constant is needed.
    f.write("\nNote: This equation uses raw NDCI values (no shift constant needed). It can be applied directly to satellite NDCI imagery.\n")

    f.write("\n\n=== Training Performance ===\n\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\n=== Test Performance ===\n\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")

# Save model
with open(os.path.join(dirs['model'], 'final_model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)


# ============================================
# 6. Residual diagnostics
# ============================================
print("\n" + "=" * 60)
print("Step 6: Residual diagnostics")
print("=" * 60)

res_train = y_train.flatten() - y_pred_train.flatten()
res_test = y_test.flatten() - y_pred_test.flatten()

# Normality test
shapiro_train = shapiro(res_train)
shapiro_test = shapiro(res_test)
print(f"  Shapiro-Wilk test:")
print(f"    Training: stat={shapiro_train.statistic:.4f}, p={shapiro_train.pvalue:.4f}")
print(f"    Test: stat={shapiro_test.statistic:.4f}, p={shapiro_test.pvalue:.4f}")

with open(os.path.join(dirs['eval'], 'residual_diagnostics.txt'), 'w', encoding='utf-8') as f:
    f.write("=== Residual Diagnostics ===\n\n")

    f.write("Shapiro-Wilk normality test:\n")
    f.write(f"  Training: stat={shapiro_train.statistic:.4f}, p-value={shapiro_train.pvalue:.4f}\n")
    f.write(f"  Test: stat={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}\n\n")

    # Breusch-Pagan test (simplified: correlation of squared residuals vs. fitted)
    bp_corr_train = pearsonr(y_pred_train.flatten(), res_train**2)
    bp_corr_test = pearsonr(y_pred_test.flatten(), res_test**2)
    f.write("Breusch-Pagan heteroscedasticity test (residual variance vs. fitted):\n")
    f.write(f"  Training: r={bp_corr_train[0]:.4f}, p={bp_corr_train[1]:.4f}\n")
    f.write(f"  Test: r={bp_corr_test[0]:.4f}, p={bp_corr_test[1]:.4f}\n\n")

    # Durbin-Watson test (simplified)
    diff_res_train = np.diff(res_train)
    dw_train = np.sum(diff_res_train**2) / np.sum(res_train**2) if np.sum(res_train**2) > 0 else np.nan
    diff_res_test = np.diff(res_test)
    dw_test = np.sum(diff_res_test**2) / np.sum(res_test**2) if np.sum(res_test**2) > 0 else np.nan
    f.write("Durbin-Watson autocorrelation test (closer to 2 is better):\n")
    f.write(f"  Training: DW={dw_train:.4f}\n")
    f.write(f"  Test: DW={dw_test:.4f}\n\n")

    # Basic residual statistics
    f.write("Residual statistics:\n")
    f.write(f"  Training: mean={np.mean(res_train):.4f}, std={np.std(res_train):.4f}\n")
    f.write(f"  Test: mean={np.mean(res_test):.4f}, std={np.std(res_test):.4f}\n")


# ============================================
# 7. Visualization
# ============================================
print("\n" + "=" * 60)
print("Step 7: Visualization")
print("=" * 60)

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
output_dir = dirs['figs']

# --- Figure 1: 4-panel ---
print("  Plotting Figure 1: Train+Test scatter and residual plots")

r_train, _ = pearsonr(y_train, y_pred_train)
r_test, _ = pearsonr(y_test, y_pred_test)

rmse_train = train_metrics['RMSE']
mae_train = train_metrics['MAE']
mre_train = train_metrics['MRE']
rmse_test = test_metrics['RMSE']
mae_test = test_metrics['MAE']
mre_test = test_metrics['MRE']

y_min = min(y_train.min(), y_test.min())
y_max = max(y_train.max(), y_test.max())

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Training scatter
axes[0].scatter(y_train, y_pred_train, alpha=0.5)
axes[0].plot([y_min, y_max], [y_min, y_max], linestyle='--', color='red')
axes[0].set_title('Train: Actual vs Predicted')
axes[0].set_xlabel(f'Actual {indicator_name} ({indicator_unit})')
axes[0].set_ylabel(f'Predicted {indicator_name} ({indicator_unit})')
axes[0].text(0.05, 0.65,
             f'r={r_train:.2f}\nRMSE={rmse_train:.2f}\nMAE={mae_train:.2f}\nMRE={mre_train:.2f}%',
             transform=axes[0].transAxes, bbox=dict(facecolor='wheat'))
axes[0].set_xlim([y_min, y_max])
axes[0].set_ylim([y_min, y_max])

# Training residuals
res_train_plot = y_train - y_pred_train
axes[1].scatter(y_pred_train, res_train_plot, alpha=0.5)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_title('Train Residuals')
axes[1].set_xlabel(f'Predicted {indicator_name} ({indicator_unit})')
axes[1].set_ylabel('Residuals')

# Test scatter
axes[2].scatter(y_test, y_pred_test, alpha=0.5)
axes[2].plot([y_min, y_max], [y_min, y_max], 'r--')
axes[2].set_title('Test: Actual vs Predicted')
axes[2].set_xlabel(f'Actual {indicator_name} ({indicator_unit})')
axes[2].set_ylabel(f'Predicted {indicator_name} ({indicator_unit})')
axes[2].text(0.05, 0.65,
             f'r={r_test:.2f}\nRMSE={rmse_test:.2f}\nMAE={mae_test:.2f}\nMRE={mre_test:.2f}%',
             transform=axes[2].transAxes, bbox=dict(facecolor='wheat'))
axes[2].set_xlim([y_min, y_max])
axes[2].set_ylim([y_min, y_max])

# Test residuals
res_test_plot = y_test - y_pred_test
axes[3].scatter(y_pred_test, res_test_plot, alpha=0.5)
axes[3].axhline(0, color='r', linestyle='--')
axes[3].set_title('Test Residuals')
axes[3].set_xlabel(f'Predicted {indicator_name} ({indicator_unit})')
axes[3].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"4.Result_{indicator_name}.png"), dpi=300)
plt.close()
print(f"    Saved: 4.Result_{indicator_name}.png (300 DPI)")

# --- Figure 2: Test-set line plot ---
print("  Plotting Figure 2: Test-set line plot")

plt.figure(figsize=(12, 4))
plt.plot(range(len(y_test)), y_test, marker='o', linestyle='-', color='g',
         label='Actual Values', linewidth=1.5, markersize=4)
plt.plot(range(len(y_pred_test)), y_pred_test, marker='x', linestyle='-', color='r',
         label='Predicted Values', linewidth=1.5, markersize=4)
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel(f'{indicator_name} ({indicator_unit})')
plt.grid(True)
plt.text(0.95, 0.95,
         f'r={r_test:.2f}\nMAE={mae_test:.2f}\nRMSE={rmse_test:.2f}\nMRE={mre_test:.2f}%',
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(facecolor='wheat'))
plt.legend(ncol=2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"5.Result_{indicator_name}_test.png"), dpi=600)
plt.close()
print(f"    Saved: 5.Result_{indicator_name}_test.png (600 DPI)")

# --- Figure S1: CV model comparison ---
print("  Plotting Figure S1: CV model comparison")
fig, ax = plt.subplots(figsize=(12, 5))
model_names = cv_df['Model'].tolist()
rmse_means = cv_df['CV_RMSE_mean'].tolist()
rmse_stds = cv_df['CV_RMSE_std'].tolist()
colors = ['#2ecc71' if m == selected_model_name else '#3498db' for m in model_names]
bars = ax.bar(model_names, rmse_means, yerr=rmse_stds, capsize=5, color=colors, alpha=0.8)
ax.set_ylabel('CV RMSE (ug/L)')
ax.set_title('Cross-Validation Performance Comparison')
ax.set_xlabel('Model')
plt.xticks(rotation=15)
ax.grid(axis='y', alpha=0.3)
# Highlight best model
best_cv_idx = cv_df['CV_RMSE_mean'].idxmin()
bars[best_cv_idx].set_color('#e74c3c')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '6.CV_model_comparison.png'), dpi=300)
plt.close()
print(f"    Saved: 6.CV_model_comparison.png (300 DPI)")

# --- Figure S2: Residual diagnostics (Q-Q plot + histogram) ---
print("  Plotting Figure S2: Residual diagnostics")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Q-Q plot
stats.probplot(res_test, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot (Test Residuals)')
axes[0].get_lines()[0].set_markersize(4)

# Histogram
axes[1].hist(res_test, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(0, color='r', linestyle='--')
axes[1].set_title('Histogram of Residuals (Test)')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '7.Residual_diagnostics.png'), dpi=300)
plt.close()
print(f"    Saved: 7.Residual_diagnostics.png (300 DPI)")


# ============================================
# 8. Final summary
# ============================================
print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nFinal model: {selected_model_name}")
print(f"Training: r={train_metrics['r']:.4f}, RMSE={train_metrics['RMSE']:.4f}, MRE={train_metrics['MRE']:.2f}%")
print(f"Test: r={test_metrics['r']:.4f}, RMSE={test_metrics['RMSE']:.4f}, MRE={test_metrics['MRE']:.2f}%")
print(f"\nGenerated files:")
for root, dirs_list, files in os.walk(OUTPUT_DIR):
    for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
        print(f"  {rel_path}")

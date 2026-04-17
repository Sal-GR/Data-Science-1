import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import boto3
import os
import gc
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import lightgbm as lgb

LOCAL_PARQUET    = "cleaned/papers.parquet"
TRAIN_YEAR_MAX   = 2009 
TEST_YEAR_MIN    = 2010
INFLUENCE_PCTILE = 75 
TFIDF_MAX_FEAT   = 20_000        
SVD_COMPONENTS   = 100 
CHUNK_SIZE       = 100_000
TFIDF_FIT_SAMPLE = 400_000 
LR_SAMPLE        = 300_000
RF_SAMPLE        = 200_000
YEAR_THRESHOLDS  = [2005, 2007, 2009, 2011, 2013]

s3 = boto3.client("s3")
os.makedirs("cleaned",                        exist_ok=True)
os.makedirs("outputs/figures/classification", exist_ok=True)
os.makedirs("outputs/data",                   exist_ok=True)

print("Downloading papers.parquet from S3 …")
s3.download_file("data-science-citation-network", "cleaned/papers.parquet", LOCAL_PARQUET)
print("  Done.")

#LOAD ONLY THE COLUMNS WE NEED 
print("\nLoading metadata columns …")
COLS = ["id", "year", "n_citation", "author_count", "reference_count", "text_combined", "venue"]
df = pd.read_parquet(LOCAL_PARQUET, columns=COLS)

# Basic cleaning
df = df.dropna(subset=["year", "n_citation"])
df["year"]            = pd.to_numeric(df["year"],            errors="coerce")
df["n_citation"]      = pd.to_numeric(df["n_citation"],      errors="coerce").fillna(0).astype(np.int32)
df["author_count"]    = pd.to_numeric(df["author_count"],    errors="coerce").fillna(1).astype(np.int16)
df["reference_count"] = pd.to_numeric(df["reference_count"], errors="coerce").fillna(0).astype(np.int16)
df["venue"]           = df["venue"].fillna("unknown").str.strip().str.lower()
df["venue"]           = df["venue"].replace("", "unknown")
df["text_combined"]   = df["text_combined"].fillna("").str.strip()
df = df[df["year"].between(1980, 2018)].reset_index(drop=True)
print(f"  Papers after cleaning: {len(df):,}")

# BUILD INFLUENCE LABEL (year relative percentile)
print("\nBuilding influence labels (year-relative percentile) …")
df["label"] = 0
year_groups = df.groupby("year")["n_citation"]
thresholds  = year_groups.transform(lambda x: x.quantile(INFLUENCE_PCTILE / 100))
df["label"] = (df["n_citation"] >= thresholds).astype(np.int8)
pos_rate = df["label"].mean()
print(f"  Influential papers: {df['label'].sum():,}  ({pos_rate:.1%} of total)")
del thresholds, year_groups
gc.collect()

#TRAIN / TEST SPLIT BY YEAR 
train_df = df[df["year"] <= TRAIN_YEAR_MAX].reset_index(drop=True)
test_df  = df[df["year"] >= TEST_YEAR_MIN].reset_index(drop=True)
print(f"\nTrain: {len(train_df):,} papers (year ≤ {TRAIN_YEAR_MAX})")
print(f"Test:  {len(test_df):,} papers  (year ≥ {TEST_YEAR_MIN})")

del df
gc.collect()

#FIT TF-IDF ON TRAIN SAMPLE ONLY (prevent vocabulary leakage)
print(f"\nFitting TF-IDF on {TFIDF_FIT_SAMPLE:,}-paper train sample …")
has_text_train = train_df[train_df["text_combined"] != ""]
tfidf_sample   = has_text_train.sample(min(TFIDF_FIT_SAMPLE, len(has_text_train)), random_state=42)

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEAT,
    min_df=5,
    max_df=0.90,
    stop_words="english",
    sublinear_tf=True,
    dtype=np.float32,
)
tfidf.fit(tfidf_sample["text_combined"])
feature_names = np.array(tfidf.get_feature_names_out())
print(f"  Vocabulary size: {len(feature_names):,}")
del tfidf_sample, has_text_train
gc.collect()

# FIT TruncatedSVD ON TRAIN SAMPLE
print(f"\nFitting TruncatedSVD ({SVD_COMPONENTS} components) on train sample …")
svd_sample = train_df[train_df["text_combined"] != ""].sample(
    min(TFIDF_FIT_SAMPLE, len(train_df)), random_state=42
)
X_svd_fit = tfidf.transform(svd_sample["text_combined"])   # sparse, safe
svd        = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42, n_iter=5)
svd.fit(X_svd_fit)
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
del X_svd_fit, svd_sample
gc.collect()

normalizer = Normalizer(copy=False)

# HELPER: build feature matrix for a DataFrame slice

def build_features(sub_df: pd.DataFrame) -> np.ndarray:
    """Return (n, SVD+3) float32 feature matrix for sub_df."""

    struct = np.column_stack([
        sub_df["author_count"].values.astype(np.float32),
        sub_df["reference_count"].values.astype(np.float32),
        (sub_df["year"].values.astype(np.float32) - 1980) / 38.0,
    ])
    X_sparse = tfidf.transform(sub_df["text_combined"].fillna(""))
    X_svd    = svd.transform(X_sparse)
    X_svd    = normalizer.transform(X_svd)
    del X_sparse
    return np.hstack([struct, X_svd]).astype(np.float32)


def build_chunked(source_df: pd.DataFrame, label: str):
    """
    Build feature matrix + labels for source_df in CHUNK_SIZE chunks.
    Concatenates at the end. Do not call this with the full 3M dataset —
    only train_df and test_df subsets should be passed.
    """
    n      = len(source_df)
    n_ch   = int(np.ceil(n / CHUNK_SIZE))
    Xs, ys = [], []
    for i in range(n_ch):
        chunk = source_df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        Xs.append(build_features(chunk))
        ys.append(chunk["label"].values.astype(np.int8))
        print(f"  [{label}] chunk {i+1}/{n_ch}  ({len(chunk):,} papers)")
        gc.collect()
    return np.vstack(Xs), np.concatenate(ys)


# BUILD TRAIN & TEST FEATURE MATRICES 
print("\nBuilding TRAIN features …")
X_train_full, y_train_full = build_chunked(train_df, "train")
print(f"  X_train shape: {X_train_full.shape}  |  positives: {y_train_full.sum():,}")

print("\nBuilding TEST features …")
X_test, y_test = build_chunked(test_df, "test")
print(f"  X_test shape:  {X_test.shape}  |  positives: {y_test.sum():,}")

gc.collect()

# SUBSAMPLE FOR MEMORY-HUNGRY MODELS 
print(f"\nSubsampling train set for LR ({LR_SAMPLE:,}) and RF ({RF_SAMPLE:,}) …")

rng = np.random.default_rng(42)
lr_idx = rng.choice(len(X_train_full), min(LR_SAMPLE, len(X_train_full)), replace=False)
rf_idx = rng.choice(len(X_train_full), min(RF_SAMPLE, len(X_train_full)), replace=False)

X_train_lr, y_train_lr = X_train_full[lr_idx], y_train_full[lr_idx]
X_train_rf, y_train_rf = X_train_full[rf_idx], y_train_full[rf_idx]
# LightGBM gets the full training data
X_train_lgb, y_train_lgb = X_train_full, y_train_full

print(f"  LR   train shape: {X_train_lr.shape}")
print(f"  RF   train shape: {X_train_rf.shape}")
print(f"  LGB  train shape: {X_train_lgb.shape}")

# TRAIN MODELS & RECORD METRICS 
results_summary = {}

# Logistic Regression
print("\n[1/3] Logistic Regression …")
t0 = time.time()
lr = LogisticRegression(
    C=1.0, max_iter=1000, solver="saga",
    class_weight="balanced", n_jobs=-1, random_state=42
)
lr.fit(X_train_lr, y_train_lr)
lr_time = time.time() - t0
y_pred_lr = lr.predict(X_test)
results_summary["Logistic Regression"] = {
    "accuracy" : accuracy_score(y_test, y_pred_lr),
    "macro_f1" : f1_score(y_test, y_pred_lr, average="macro"),
    "train_sec": lr_time,
}
print(f"  Accuracy: {results_summary['Logistic Regression']['accuracy']:.4f}  "
      f"Macro-F1: {results_summary['Logistic Regression']['macro_f1']:.4f}  "
      f"Time: {lr_time:.1f}s")

# 9b. Random Forest
print("\n[2/3] Random Forest …")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf.fit(X_train_rf, y_train_rf)
rf_time = time.time() - t0
y_pred_rf = rf.predict(X_test)
results_summary["Random Forest"] = {
    "accuracy" : accuracy_score(y_test, y_pred_rf),
    "macro_f1" : f1_score(y_test, y_pred_rf, average="macro"),
    "train_sec": rf_time,
}
print(f"  Accuracy: {results_summary['Random Forest']['accuracy']:.4f}  "
      f"Macro-F1: {results_summary['Random Forest']['macro_f1']:.4f}  "
      f"Time: {rf_time:.1f}s")

# Free RF immediately, it is the biggest memory object at 1–2 GB
del X_train_rf, y_train_rf
gc.collect()

# 9c. LightGBM
print("\n[3/3] LightGBM …")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    max_depth=-1, min_child_samples=50,
    class_weight="balanced", n_jobs=-1, random_state=42,
    verbosity=-1,
)
lgb_model.fit(
    X_train_lgb, y_train_lgb,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
)
lgb_time = time.time() - t0
y_pred_lgb = lgb_model.predict(X_test)
results_summary["LightGBM"] = {
    "accuracy" : accuracy_score(y_test, y_pred_lgb),
    "macro_f1" : f1_score(y_test, y_pred_lgb, average="macro"),
    "train_sec": lgb_time,
}
print(f"  Accuracy: {results_summary['LightGBM']['accuracy']:.4f}  "
      f"Macro-F1: {results_summary['LightGBM']['macro_f1']:.4f}  "
      f"Time: {lgb_time:.1f}s")

del X_train_lgb, y_train_lgb, X_train_lr, y_train_lr, X_train_full, y_train_full
gc.collect()

# FIGURE 1: Model Comparison Bar Chart 
print("\nSaving Figure 1: model comparison …")
models   = list(results_summary.keys())
accuracy = [results_summary[m]["accuracy"] for m in models]
macro_f1 = [results_summary[m]["macro_f1"] for m in models]
times    = [results_summary[m]["train_sec"] for m in models]

x     = np.arange(len(models))
width = 0.35
colors_bar = ["#2563eb", "#16a34a", "#dc2626"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy + Macro-F1
ax = axes[0]
b1 = ax.bar(x - width/2, accuracy, width, label="Accuracy", color=colors_bar, alpha=0.85, edgecolor="white")
b2 = ax.bar(x + width/2, macro_f1,  width, label="Macro-F1", color=colors_bar, alpha=0.45, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Model Accuracy & Macro-F1\n(temporal split: train ≤ 2009, test ≥ 2010)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)

# Training time
ax2 = axes[1]
ax2.bar(models, times, color=colors_bar, alpha=0.85, edgecolor="white")
ax2.set_ylabel("Training Time (seconds)")
ax2.set_title("Training Time Comparison", fontsize=11, fontweight="bold")
for i, t in enumerate(times):
    ax2.text(i, t + max(times)*0.01, f"{t:.1f}s", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/classification/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 2: Confusion Matrix for Best Model 
print("Saving Figure 2: confusion matrix …")
best_name  = max(results_summary, key=lambda m: results_summary[m]["macro_f1"])
best_preds = {"Logistic Regression": y_pred_lr,
              "Random Forest":       y_pred_rf,
              "LightGBM":            y_pred_lgb}[best_name]

cm_mat = confusion_matrix(y_test, best_preds)
disp   = ConfusionMatrixDisplay(confusion_matrix=cm_mat,
                                display_labels=["Not Influential", "Influential"])
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}\n(test set: year ≥ {TEST_YEAR_MIN})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/classification/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 3: Top Predictive Keywords (Logistic Regression) 
print("Saving Figure 3: top predictive keywords …")
N_TOP       = 20
print("  Fitting sparse LR on 200k-sample for keyword extraction …")
kw_sample = train_df[train_df["text_combined"] != ""].sample(
    min(200_000, len(train_df[train_df["text_combined"] != ""])), random_state=7
)
X_kw_sparse = tfidf.transform(kw_sample["text_combined"])
y_kw        = kw_sample["label"].values

kw_lr = LogisticRegression(
    C=0.5, max_iter=500, solver="saga",
    class_weight="balanced", n_jobs=-1, random_state=42
)
kw_lr.fit(X_kw_sparse, y_kw)
del X_kw_sparse, kw_sample
gc.collect()

coefs = kw_lr.coef_[0]   # shape: (vocab,)
top_pos_idx  = coefs.argsort()[-N_TOP:][::-1]
top_neg_idx  = coefs.argsort()[:N_TOP]

top_pos_words  = feature_names[top_pos_idx]
top_pos_coefs  = coefs[top_pos_idx]
top_neg_words  = feature_names[top_neg_idx]
top_neg_coefs  = coefs[top_neg_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(top_pos_words[::-1], top_pos_coefs[::-1], color="#16a34a", edgecolor="white")
axes[0].set_title(f"Top {N_TOP} Keywords → Influential\n(positive LR coefficients)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Coefficient")

axes[1].barh(top_neg_words[::-1], top_neg_coefs[::-1], color="#dc2626", edgecolor="white")
axes[1].set_title(f"Top {N_TOP} Keywords → Not Influential\n(negative LR coefficients)",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Coefficient")

plt.suptitle("Logistic Regression — Most Predictive Keywords", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/classification/top_keywords.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 4: LightGBM Feature Importance 
print("Saving Figure 4: LightGBM feature importance …")
struct_names = ["author_count", "reference_count", "year_norm"]
all_names    = struct_names + [f"svd_{i}" for i in range(SVD_COMPONENTS)]

importances = lgb_model.feature_importances_
# Show only top 20 SVD + all 3 structured features for readability
top_feat_idx = np.argsort(importances)[-23:][::-1]   # 20 SVD + 3 struct
top_feat_names  = [all_names[i] if i < len(all_names) else f"feat_{i}" for i in top_feat_idx]
top_feat_scores = importances[top_feat_idx]

fig, ax = plt.subplots(figsize=(10, 7))
colors_fi = ["#f59e0b" if n in struct_names else "#6366f1" for n in top_feat_names]
ax.barh(top_feat_names[::-1], top_feat_scores[::-1], color=colors_fi[::-1], edgecolor="white")
ax.set_title("LightGBM Feature Importance (top 23)\n"
             "Orange = structured feature  |  Purple = SVD text component",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Importance (split count)")
plt.tight_layout()
plt.savefig("outputs/figures/classification/lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 5: Performance Degradation Across Year Thresholds
print("\nRunning degradation experiment across year thresholds …")
print("  Thresholds:", YEAR_THRESHOLDS)

FULL_DF = pd.concat([train_df, test_df], ignore_index=True)
del train_df, test_df
gc.collect()

degrad_f1s        = []
degrad_train_sizes = []

for thresh in YEAR_THRESHOLDS:
    tr = FULL_DF[FULL_DF["year"] <= thresh].reset_index(drop=True)
    te = FULL_DF[FULL_DF["year"] >  thresh].reset_index(drop=True)
    if len(tr) == 0 or len(te) == 0:
        degrad_f1s.append(np.nan)
        degrad_train_sizes.append(0)
        continue

    # Subsample so each run stays ~150k train, 100k test
    tr_s = tr.sample(min(150_000, len(tr)), random_state=42)
    te_s = te.sample(min(100_000, len(te)), random_state=42)

    X_tr, y_tr = build_features(tr_s), tr_s["label"].values.astype(np.int8)
    X_te, y_te = build_features(te_s), te_s["label"].values.astype(np.int8)

    dg_model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1
    )
    dg_model.fit(X_tr, y_tr)
    f1 = f1_score(y_te, dg_model.predict(X_te), average="macro")
    degrad_f1s.append(f1)
    degrad_train_sizes.append(len(tr_s))
    print(f"  threshold={thresh}  train={len(tr_s):,}  test={len(te_s):,}  macro-F1={f1:.4f}")

    del tr, te, tr_s, te_s, X_tr, y_tr, X_te, y_te, dg_model
    gc.collect()

del FULL_DF
gc.collect()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(YEAR_THRESHOLDS, degrad_f1s, "o-", color="#2563eb", linewidth=2.5, markersize=8, label="Macro-F1")
ax1.set_xlabel("Train / Test Split Year", fontsize=11)
ax1.set_ylabel("Macro-F1 on Test Set", color="#2563eb", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#2563eb")
ax1.set_ylim(0, 1)
for x, y in zip(YEAR_THRESHOLDS, degrad_f1s):
    if not np.isnan(y):
        ax1.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=9, color="#2563eb")

ax2 = ax1.twinx()
ax2.bar(YEAR_THRESHOLDS, degrad_train_sizes, width=1.2, alpha=0.18, color="#9ca3af", label="Train size")
ax2.set_ylabel("Train Set Size", color="#9ca3af", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#9ca3af")

lines1, lbs1 = ax1.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9, loc="lower right")
ax1.set_title("Model Performance vs. Train/Test Year Threshold\n"
              "(LightGBM, 150k train sample, 100k test sample each)",
              fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/classification/degradation_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# PRINT SUMMARY
print("\n" + "=" * 60)
print("TEMPORAL CLASSIFICATION SUMMARY")
print("=" * 60)
print(f"Train: year ≤ {TRAIN_YEAR_MAX}  |  Test: year ≥ {TEST_YEAR_MIN}")
print(f"Influence threshold: top {INFLUENCE_PCTILE}th percentile within each year cohort")
print(f"Features: {SVD_COMPONENTS} SVD text dims + 3 structured = {SVD_COMPONENTS + 3} total\n")
print(f"{'Model':<22}  {'Accuracy':>8}  {'Macro-F1':>8}  {'Train Time':>10}")
print("-" * 56)
for m, v in results_summary.items():
    print(f"  {m:<20}  {v['accuracy']:>8.4f}  {v['macro_f1']:>8.4f}  {v['train_sec']:>8.1f}s")
print(f"\nBest model by Macro-F1: {best_name}")
print("\nDegradation experiment (LightGBM Macro-F1 by threshold):")
for t, f in zip(YEAR_THRESHOLDS, degrad_f1s):
    print(f"  Split year {t}: {f:.4f}" if not np.isnan(f) else f"  Split year {t}: N/A")
print("\nFigures → outputs/figures/classification/")

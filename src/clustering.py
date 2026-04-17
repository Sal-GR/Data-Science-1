import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import boto3
import os
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

CHUNK_SIZE     = 100_000    # rows per chunk for processing (adjust based on RAM)
TFIDF_SAMPLE   = 500_000    # papers used to fit TF-IDF vocabulary
N_COMPONENTS   = 100        # SVD dimensions for clustering
K_VALUES       = [5, 8, 10, 12, 15, 20]
SCATTER_SAMPLE = 50_000     # points in 2D scatter plot
LOCAL_PARQUET  = "cleaned/papers.parquet"

s3 = boto3.client("s3")
os.makedirs("cleaned", exist_ok=True)
os.makedirs("outputs/figures/clustering", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)

s3.download_file("data-science-citation-network", "cleaned/papers.parquet", LOCAL_PARQUET)
print("Downloaded papers.parquet from S3")

# 2. Fit TF-IDF vocabulary on a streaming sample
#    Read only TFIDF_SAMPLE rows into RAM 
#    never load the full 3M DataFrame at once
print(f"\nReading {TFIDF_SAMPLE:,}-paper sample for TF-IDF fit...")
sample_df = pd.read_parquet(
    LOCAL_PARQUET,
    columns=["text_combined"]
).dropna(subset=["text_combined"])
sample_df = sample_df[sample_df["text_combined"].str.strip() != ""]
sample_df = sample_df.sample(min(TFIDF_SAMPLE, len(sample_df)), random_state=42)

tfidf = TfidfVectorizer(
    max_features=15_000,    
    min_df=5,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True,
    dtype=np.float32        
)
tfidf.fit(sample_df["text_combined"])
feature_names = np.array(tfidf.get_feature_names_out())
print(f"Vocabulary size: {len(feature_names):,}")
del sample_df               # free ~1–2 GB immediately

# 3. Count total valid rows without loading all
#    Used to preallocate result arrays and
#    track progress accurately across chunks
print("\nCounting valid papers...")
id_text = pd.read_parquet(LOCAL_PARQUET, columns=["id", "text_combined"])
id_text = id_text[id_text["text_combined"].notna() & (id_text["text_combined"].str.strip() != "")]
id_text = id_text.reset_index(drop=True)
total_papers = len(id_text)
n_chunks = int(np.ceil(total_papers / CHUNK_SIZE))
print(f"Valid papers: {total_papers:,}  |  Chunks: {n_chunks}")

# 4. PASS 1 — Fit TruncatedSVD chunk by chunk
print(f"\nPass 1: Fitting TruncatedSVD ({N_COMPONENTS} components) in {n_chunks} chunks...")

# Fit SVD on a large sample: TruncatedSVD does not
# have partial_fit, so we use a sample based fit.
# 500k papers in sparse format is only ~200–400 MB.
svd_fit_sample = id_text.sample(min(500_000, total_papers), random_state=42)
X_svd_fit = tfidf.transform(svd_fit_sample["text_combined"])   # sparse, safe
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42, n_iter=5)
svd.fit(X_svd_fit)
print(f"Explained variance ({N_COMPONENTS} components): {svd.explained_variance_ratio_.sum():.2%}")
del X_svd_fit, svd_fit_sample

# Normalizer improves cluster quality after SVD
normalizer = Normalizer(copy=False)

# 5. Choose best k on a sample
#    Transform a 100k sample through SVD,
#    then evaluate each k, all sparse, safe
eval_n = min(100_000, total_papers)
print(f"\nEvaluating k values on {eval_n:,}-paper sample...")
eval_sample  = id_text.sample(eval_n, random_state=99)
X_eval_svd   = svd.transform(tfidf.transform(eval_sample["text_combined"]))
X_eval_svd   = normalizer.transform(X_eval_svd)
del eval_sample

inertias    = []
silhouettes = []

for k in K_VALUES:
    model  = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10_000, n_init=3)
    labels = model.fit_predict(X_eval_svd)
    inertias.append(model.inertia_)
    sil_idx = np.random.choice(len(X_eval_svd), min(10_000, len(X_eval_svd)), replace=False)
    sil     = silhouette_score(X_eval_svd[sil_idx], labels[sil_idx], random_state=42)
    silhouettes.append(sil)
    print(f"  k={k:2d}  inertia={model.inertia_:.0f}  silhouette={sil:.4f}")

del X_eval_svd

# Elbow + Silhouette plot 
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(K_VALUES, inertias,    "o-",  color="#2563eb", linewidth=2, label="Inertia")
ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
ax1.set_ylabel("Inertia", color="#2563eb", fontsize=12)
ax1.tick_params(axis="y", labelcolor="#2563eb")
ax2 = ax1.twinx()
ax2.plot(K_VALUES, silhouettes, "s--", color="#dc2626", linewidth=2, label="Silhouette")
ax2.set_ylabel("Silhouette Score (sample)", color="#dc2626", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#dc2626")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.title("Elbow & Silhouette — Choosing k", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/clustering/elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()

best_k = K_VALUES[int(np.argmax(silhouettes))]
print(f"\nBest k by silhouette: {best_k}")

# 6. PASS 2, Fit MiniBatchKMeans on full data
#    Each chunk: sparse tfidf, svd, normalizer
#    Peak RAM per chunk: ~50–200 MB (sparse)
#    + ~100k x 100 floats (~40 MB dense SVD output)
print(f"\nPass 2: Fitting MiniBatchKMeans (k={best_k}) on full dataset...")
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=10_000, n_init=5)

for i in range(n_chunks):
    chunk_ids = id_text.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse  = tfidf.transform(chunk_ids["text_combined"])   # sparse
    X_svd     = svd.transform(X_sparse)                       # dense but tiny: 100k x 100
    X_svd     = normalizer.transform(X_svd)
    kmeans.partial_fit(X_svd)
    print(f"  KMeans fit chunk {i+1}/{n_chunks}  ({len(chunk_ids):,} papers)")
    del X_sparse, X_svd

print(f"\nPass 3: Assigning labels and collecting results...")

all_ids     = []
all_labels  = []
all_pca_x   = []
all_pca_y   = []
all_years   = []
all_cites   = []

# Load only the columns we need for result collection
meta_df = pd.read_parquet(LOCAL_PARQUET, columns=["id", "year", "n_citation", "text_combined"])
meta_df = meta_df[meta_df["text_combined"].notna() & (meta_df["text_combined"].str.strip() != "")]
meta_df = meta_df.reset_index(drop=True)

for i in range(n_chunks):
    chunk    = meta_df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse = tfidf.transform(chunk["text_combined"])
    X_svd    = svd.transform(X_sparse)
    X_norm   = normalizer.transform(X_svd)
    labels   = kmeans.predict(X_norm)

    all_ids.append(chunk["id"].values)
    all_labels.append(labels)
    all_pca_x.append(X_svd[:, 0])
    all_pca_y.append(X_svd[:, 1])   
    all_years.append(chunk["year"].values)
    all_cites.append(chunk["n_citation"].values)

    print(f"  Labels assigned chunk {i+1}/{n_chunks}")
    del X_sparse, X_svd, X_norm, labels

del meta_df

# Assemble results DataFrame: only what we need for plots + output
results = pd.DataFrame({
    "id"         : np.concatenate(all_ids),
    "cluster"    : np.concatenate(all_labels),
    "pca_x"      : np.concatenate(all_pca_x),
    "pca_y"      : np.concatenate(all_pca_y),
    "year"       : np.concatenate(all_years),
    "n_citation" : np.concatenate(all_cites),
})
del all_ids, all_labels, all_pca_x, all_pca_y, all_years, all_cites
print(f"Results assembled: {len(results):,} papers")

# 8. Label clusters by top TF-IDF terms
print("\nExtracting top terms per cluster...")
cluster_top_terms = {}

for c in range(best_k):
    idx       = np.where(results["cluster"] == c)[0][:3000]
    texts     = id_text.iloc[idx]["text_combined"]
    X_sample  = tfidf.transform(texts)
    centroid  = np.asarray(X_sample.mean(axis=0)).flatten()
    top_idx   = centroid.argsort()[-5:][::-1]
    top_words = ", ".join(feature_names[top_idx])
    cluster_top_terms[c] = f"C{c}: {top_words}"
    print(f"  Cluster {c}: {top_words}")

del id_text
colors = cm.tab20(np.linspace(0, 1, best_k))

# 9. Figure 1: 2D Cluster Scatter
plot_sample = results.sample(min(SCATTER_SAMPLE, len(results)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 8))
for c in range(best_k):
    mask = plot_sample["cluster"] == c
    ax.scatter(
        plot_sample.loc[mask, "pca_x"],
        plot_sample.loc[mask, "pca_y"],
        s=1, alpha=0.4, color=colors[c],
        label=cluster_top_terms[c]
    )
ax.set_title(
    f"Paper Clusters (k={best_k}) — SVD 2D Projection\n({SCATTER_SAMPLE:,} paper sample)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("SVD Component 1")
ax.set_ylabel("SVD Component 2")
ax.legend(loc="upper right", fontsize=6, markerscale=5,
          bbox_to_anchor=(1.38, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig("outputs/figures/clustering/cluster_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# 10. Figure 2: Cluster sizes
cluster_counts = results["cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    [cluster_top_terms[c] for c in cluster_counts.index],
    cluster_counts.values,
    color=[colors[c] for c in cluster_counts.index],
    edgecolor="white"
)
ax.set_title("Number of Papers per Cluster", fontsize=14, fontweight="bold")
ax.set_ylabel("Paper Count")
ax.set_xlabel("Cluster")
plt.xticks(rotation=45, ha="right", fontsize=7)
for bar, val in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/clustering/cluster_sizes.png", dpi=150, bbox_inches="tight")
plt.close()

# 11. Figure 3: Median citations per cluster
median_citations = results.groupby("cluster")["n_citation"].median().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    [cluster_top_terms[c] for c in median_citations.index],
    median_citations.values,
    color=[colors[c] for c in median_citations.index],
    edgecolor="white"
)
ax.set_title("Median Citations per Cluster", fontsize=14, fontweight="bold")
ax.set_ylabel("Median n_citation")
ax.set_xlabel("Cluster")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/clustering/cluster_median_citations.png", dpi=150, bbox_inches="tight")
plt.close()

# 12. Figure 4: Cluster growth over time
results["decade"] = (results["year"] // 10 * 10).astype(int)
pivot = results.groupby(["decade", "cluster"]).size().unstack(fill_value=0)
pivot = pivot.loc[(pivot.index >= 1980) & (pivot.index <= 2020)]

fig, ax = plt.subplots(figsize=(12, 6))
for c in range(best_k):
    if c in pivot.columns:
        ax.plot(pivot.index, pivot[c], marker="o",
                label=cluster_top_terms[c], color=colors[c], linewidth=1.5)
ax.set_title("Cluster Growth Over Time (by Decade)", fontsize=14, fontweight="bold")
ax.set_xlabel("Decade")
ax.set_ylabel("Number of Papers")
ax.legend(fontsize=6, bbox_to_anchor=(1.38, 1), loc="upper right", borderaxespad=0)
plt.tight_layout()
plt.savefig("outputs/figures/clustering/cluster_growth_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# 13. Save cluster mapping for Network.py
cluster_map = results[["id", "cluster"]].copy()
cluster_map.to_parquet("outputs/data/paper_clusters.parquet", index=False)
print(f"\nCluster mapping saved: {len(cluster_map):,} papers → outputs/data/paper_clusters.parquet")
print("Clustering complete. Figures saved to outputs/figures/clustering/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import boto3
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

CHUNK_SIZE      = 80_000   # size is what works for low memory
TFIDF_SAMPLE    = 500_000   # papers used to fit TF-IDF vocabulary
N_COMPONENTS    = 50        # PCA dimensions for clustering
N_COMPONENTS_2D = 2         # PCA dimensions for visualization
K_VALUES        = [5, 8, 10, 12, 15, 20]
SCATTER_SAMPLE  = 50_000    # points to plot in scatter (3M is unreadable)

s3 = boto3.client("s3")
os.makedirs("cleaned", exist_ok=True)
s3.download_file(
    "data-science-citation-network",
    "cleaned/papers.parquet",
    "cleaned/papers.parquet"
)
df = pd.read_parquet("cleaned/papers.parquet")
os.makedirs("outputs/figures/clustering", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)

# 1. Drop papers with no text
df = df[df["text_combined"].notna() & (df["text_combined"].str.strip() != "")].copy()
df = df.reset_index(drop=True)
print(f"Papers with text: {len(df):,}")

# 2. Fit TF-IDF on a representative sample
#    fit ONCE on a sample so that vocabulary stays consistent across all chunks
print(f"\nFitting TF-IDF on {TFIDF_SAMPLE:,}-paper sample...")
tfidf_sample = df.sample(min(TFIDF_SAMPLE, len(df)), random_state=42)

tfidf = TfidfVectorizer(
    max_features=20_000,
    min_df=5,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True
)
tfidf.fit(tfidf_sample["text_combined"])
feature_names = np.array(tfidf.get_feature_names_out())
print(f"Vocabulary size: {len(feature_names):,}")
del tfidf_sample  # free memory

n_chunks = int(np.ceil(len(df) / CHUNK_SIZE))

# 3. PASS 1: Fit IncrementalPCA chunk by chunk
#    Each chunk is ~100k x 20k dense float32
#    (~8GB), never loading the full 3M at once
print(f"\nPass 1: Fitting IncrementalPCA ({N_COMPONENTS} components) in {n_chunks} chunks...")
ipca = IncrementalPCA(n_components=N_COMPONENTS)

for i in range(n_chunks):
    chunk    = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse = tfidf.transform(chunk["text_combined"])
    X_dense  = X_sparse.toarray().astype(np.float32)
    ipca.partial_fit(X_dense)
    print(f"  PCA fit chunk {i+1}/{n_chunks}  ({len(chunk):,} papers)")
    del X_sparse, X_dense

print(f"Explained variance (50 components): {ipca.explained_variance_ratio_.sum():.2%}")

# 4. Choose best k: Elbow + Silhouette
#    Evaluated on a manageable sample so we
#    don't need another full pass just for this
eval_n = min(100_000, len(df))
print(f"\nEvaluating k values on a {eval_n:,}-paper sample...")
eval_sample  = df.sample(eval_n, random_state=99)
X_eval_pca   = ipca.transform(
    tfidf.transform(eval_sample["text_combined"]).toarray().astype(np.float32)
)
del eval_sample

inertias    = []
silhouettes = []

for k in K_VALUES:
    model  = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10_000, n_init=3)
    labels = model.fit_predict(X_eval_pca)
    inertias.append(model.inertia_)
    sil_idx = np.random.choice(len(X_eval_pca), min(10_000, len(X_eval_pca)), replace=False)
    sil     = silhouette_score(X_eval_pca[sil_idx], labels[sil_idx], random_state=42)
    silhouettes.append(sil)
    print(f"  k={k:2d}  inertia={model.inertia_:.0f}  silhouette={sil:.4f}")

del X_eval_pca

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

# 5. PASS 2: Fit final MiniBatchKMeans
#    partial_fit() streams all 3M papers without holding them all in memory at once
print(f"\nPass 2: Fitting MiniBatchKMeans (k={best_k}) on full dataset...")
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=10_000, n_init=5)

for i in range(n_chunks):
    chunk    = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse = tfidf.transform(chunk["text_combined"])
    X_pca    = ipca.transform(X_sparse.toarray().astype(np.float32))
    kmeans.partial_fit(X_pca)
    print(f"  KMeans fit chunk {i+1}/{n_chunks}")
    del X_sparse, X_pca

# 6. PASS 3: Assign labels + collect 2D coords
#    Fit a second 2D IncrementalPCA just for
#    the scatter plot visualization
print(f"\nPass 3a: Fitting 2D IncrementalPCA for visualization...")
ipca_2d = IncrementalPCA(n_components=N_COMPONENTS_2D)

for i in range(n_chunks):
    chunk    = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse = tfidf.transform(chunk["text_combined"])
    X_pca    = ipca.transform(X_sparse.toarray().astype(np.float32))
    ipca_2d.partial_fit(X_pca)
    del X_sparse, X_pca

print(f"Pass 3b: Assigning cluster labels and 2D coords...")
all_labels = []
all_pca_x  = []
all_pca_y  = []

for i in range(n_chunks):
    chunk    = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    X_sparse = tfidf.transform(chunk["text_combined"])
    X_pca    = ipca.transform(X_sparse.toarray().astype(np.float32))
    labels   = kmeans.predict(X_pca)
    coords   = ipca_2d.transform(X_pca)
    all_labels.append(labels)
    all_pca_x.append(coords[:, 0])
    all_pca_y.append(coords[:, 1])
    print(f"  Labels assigned chunk {i+1}/{n_chunks}")
    del X_sparse, X_pca, labels, coords

df["cluster"] = np.concatenate(all_labels)
df["pca_x"]   = np.concatenate(all_pca_x)
df["pca_y"]   = np.concatenate(all_pca_y)
del all_labels, all_pca_x, all_pca_y

# 7. Label each cluster by its top TF-IDF terms
print("\nExtracting top terms per cluster...")
cluster_top_terms = {}

for c in range(best_k):
    idx        = np.where(df["cluster"] == c)[0][:5000]   # sample for speed
    X_sample   = tfidf.transform(df.iloc[idx]["text_combined"])
    centroid   = np.asarray(X_sample.mean(axis=0)).flatten()
    top_idx    = centroid.argsort()[-5:][::-1]
    top_words  = ", ".join(feature_names[top_idx])
    cluster_top_terms[c] = f"C{c}: {top_words}"
    print(f"  Cluster {c}: {top_words}")

colors = cm.tab20(np.linspace(0, 1, best_k))

# 8. Figure 1: 2D Cluster Scatter
plot_sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=42)

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
    f"Paper Clusters (k={best_k}) — PCA 2D Projection\n({SCATTER_SAMPLE:,} paper sample)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend(loc="upper right", fontsize=6, markerscale=5,
          bbox_to_anchor=(1.38, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig("outputs/figures/clustering/cluster_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Figure 2: Cluster sizes
cluster_counts = df["cluster"].value_counts().sort_index()

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

# 10. Figure 3: Median citations per cluster
median_citations = df.groupby("cluster")["n_citation"].median().sort_values(ascending=False)

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

# 11. Figure 4: Cluster growth over time
df["decade"] = (df["year"] // 10 * 10).astype(int)
pivot = df.groupby(["decade", "cluster"]).size().unstack(fill_value=0)
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

# 12. Save cluster mapping for Network.py
#     Network.py loads this to color nodes
#     by research area in the citation graph
cluster_map = df[["id", "cluster"]].copy()
cluster_map.to_parquet("outputs/data/paper_clusters.parquet", index=False)
print(f"\nCluster mapping saved: {len(cluster_map):,} papers → outputs/data/paper_clusters.parquet")
print("Clustering complete. Figures saved to outputs/figures/clustering/")
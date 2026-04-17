import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import boto3
import os
import gc

import networkx as nx
import pyarrow.parquet as pq

LOCAL_PARQUET        = "cleaned/papers.parquet"
LOCAL_CLUSTER        = "cleaned/paper_clusters.parquet"
PAGERANK_ALPHA       = 0.85
SUBGRAPH_TOP_N       = 300
CROSS_CLUSTER_SAMPLE = 200_000

s3 = boto3.client("s3")
os.makedirs("cleaned",                 exist_ok=True)
os.makedirs("outputs/figures/network", exist_ok=True)
os.makedirs("outputs/data",            exist_ok=True)

print("Downloading papers.parquet …")
s3.download_file("data-science-citation-network", "cleaned/papers.parquet", LOCAL_PARQUET)

print("Downloading paper_clusters.parquet …")
try:
    s3.download_file(
        "data-science-citation-network",
        "outputs/data/paper_clusters.parquet",
        LOCAL_CLUSTER
    )
except Exception:
    local_fallback = "outputs/data/paper_clusters.parquet"
    if os.path.exists(local_fallback):
        import shutil
        shutil.copy(local_fallback, LOCAL_CLUSTER)
        print("  Using locally produced paper_clusters.parquet")
    else:
        raise FileNotFoundError(
            "paper_clusters.parquet not found. "
            "Ensure clustering.py runs before Network.py in FilesToRun.txt."
        )

print("\nLoading paper metadata …")
meta = pd.read_parquet(
    LOCAL_PARQUET,
    columns=["id", "title", "year", "n_citation"]
).dropna(subset=["id"])
meta["year"]       = pd.to_numeric(meta["year"],       errors="coerce")
meta["n_citation"] = pd.to_numeric(meta["n_citation"], errors="coerce").fillna(0).astype(int)

clusters = pd.read_parquet(LOCAL_CLUSTER, columns=["id", "cluster"])

#  DIAGNOSTIC: check join health before proceeding 
print(f"\n  papers.parquet IDs  — count: {len(meta):,}  sample: {meta['id'].iloc[0]!r}")
print(f"  paper_clusters IDs  — count: {len(clusters):,}  sample: {clusters['id'].iloc[0]!r}")
overlap = len(set(meta["id"]) & set(clusters["id"]))
print(f"  ID overlap: {overlap:,} ({overlap/len(meta)*100:.1f}% of papers have a cluster label)")

meta = meta.merge(clusters, on="id", how="left")
meta["cluster"] = meta["cluster"].fillna(-1).astype(np.int16)
labelled_count  = (meta["cluster"] >= 0).sum()
print(f"  After merge: {labelled_count:,} papers have a cluster label ({labelled_count/len(meta)*100:.1f}%)")
del clusters
gc.collect()

valid_ids     = set(meta["id"])
id_to_cluster = dict(zip(meta["id"], meta["cluster"]))
total_papers  = len(meta)
print(f"  Total papers: {total_papers:,}")

#Build DiGraph using pyarrow row groups
print("\nBuilding DiGraph via pyarrow row groups …")
G           = nx.DiGraph()
G.add_nodes_from(valid_ids)
total_edges = 0

pf          = pq.ParquetFile(LOCAL_PARQUET)
n_groups    = pf.metadata.num_row_groups
print(f"  Parquet has {n_groups} row groups")

#  DIAGNOSTIC: inspect the references column in the first row group 
rg0     = pf.read_row_group(0, columns=["id", "references"]).to_pandas()
ref_col = rg0["references"]
print(f"\n  references dtype   : {ref_col.dtype}")
print(f"  references[0] type : {type(ref_col.iloc[0])}")
print(f"  references[0] value: {ref_col.iloc[0]!r}")
non_null = ref_col.dropna()
non_empty = non_null[non_null.apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)]
print(f"  Non-null refs in rg0: {len(non_null):,}  |  Non-empty lists: {len(non_empty):,}")
if len(non_empty) > 0:
    print(f"  First non-empty ref list: {non_empty.iloc[0]!r}")
del rg0, ref_col, non_null, non_empty

for g in range(n_groups):
    rg = pf.read_row_group(g, columns=["id", "references"]).to_pandas()

    def to_list(x):
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        if isinstance(x, str):
            import ast
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []

    rg["references"] = rg["references"].apply(to_list)

    rg = rg[rg["references"].map(len) > 0]
    if len(rg) == 0:
        del rg
        continue

    exploded = rg.explode("references").dropna(subset=["references"])
    exploded = exploded.rename(columns={"id": "citing", "references": "cited"})

    mask  = exploded["cited"].isin(valid_ids)
    edges = list(zip(exploded.loc[mask, "citing"], exploded.loc[mask, "cited"]))

    G.add_edges_from(edges)
    total_edges += len(edges)
    del rg, exploded, edges
    gc.collect()

    if (g + 1) % 5 == 0 or g == n_groups - 1:
        print(f"  Row group {g+1}/{n_groups}  |  edges so far: {total_edges:,}")

print(f"\nGraph built: {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")

#DIAGNOSTIC: abort early with clear message if graph is still empty 
if G.number_of_edges() == 0:
    raise RuntimeError(
        "Graph has 0 edges after processing all row groups.\n"
        "This means the 'references' column could not be parsed into valid IDs.\n"
        "Check the diagnostic output above — look at 'references[0] value' and "
        "'references dtype' to see what the column actually contains."
    )

# PageRank
print("\nComputing PageRank …")
try:
    pagerank = nx.pagerank_scipy(G, alpha=PAGERANK_ALPHA, max_iter=200, tol=1e-6)
    print("  Used pagerank_scipy")
except AttributeError:
    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, max_iter=200, tol=1e-6)
    print("  Used nx.pagerank")

in_degree = dict(G.in_degree())

# Check if PageRank is still uniform the solver failed
pr_values = list(pagerank.values())
print(f"  PageRank — min: {min(pr_values):.3e}  max: {max(pr_values):.3e}  "
      f"ratio: {max(pr_values)/max(min(pr_values), 1e-20):.1f}x")

# Extract subgraph before freeing G
print(f"Extracting top-{SUBGRAPH_TOP_N} subgraph …")
meta["pagerank"]  = meta["id"].map(pagerank)
meta["in_degree"] = meta["id"].map(in_degree).fillna(0).astype(int)
top_ids = set(meta.nlargest(SUBGRAPH_TOP_N, "pagerank")["id"])
SG      = G.subgraph(top_ids).copy()

del G
gc.collect()
print("  Full graph freed")

# Save
out_cols = ["id", "title", "year", "n_citation", "cluster", "pagerank", "in_degree"]
meta[out_cols].to_parquet("outputs/data/network_metrics.parquet", index=False)
print("Saved outputs/data/network_metrics.parquet")

# Color map
unique_clusters = sorted(c for c in meta["cluster"].unique() if c >= 0)
n_clusters      = len(unique_clusters)
cmap            = cm.get_cmap("tab20", max(n_clusters, 1))
c_to_idx        = {c: i for i, c in enumerate(unique_clusters)}

def cluster_color(c):
    return cmap(c_to_idx.get(int(c), 0))

# Figure 1: Top-20 papers by PageRank 
top20 = meta.nlargest(20, "pagerank")[
    ["title", "year", "n_citation", "cluster", "pagerank"]
].copy()
top20["short_title"] = top20["title"].str[:55] + "…"

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(
    top20["short_title"][::-1],
    top20["pagerank"][::-1],
    color=[cluster_color(c) for c in top20["cluster"][::-1]],
    edgecolor="white"
)
ax.set_xlabel("PageRank Score", fontsize=11)
ax.set_title("Top 20 Papers by PageRank\n(colored by cluster)", fontsize=13, fontweight="bold")
for bar, (_, row) in zip(bars[::-1], top20.iterrows()):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"  yr={int(row.year)}  cites={int(row.n_citation):,}  C{int(row.cluster)}",
            va="center", fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/network/top20_pagerank.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved top20_pagerank.png")

# Figure 2: PageRank vs citation count 
scatter = meta.dropna(subset=["pagerank"])
scatter = scatter[scatter["n_citation"] > 0].sample(min(80_000, len(scatter)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
for c in unique_clusters:
    mask = scatter["cluster"] == c
    ax.scatter(
        np.log1p(scatter.loc[mask, "n_citation"]),
        scatter.loc[mask, "pagerank"],
        s=3, alpha=0.3, color=cluster_color(c), label=f"C{c}"
    )
ax.set_xlabel("log(1 + n_citation)", fontsize=11)
ax.set_ylabel("PageRank Score", fontsize=11)
ax.set_title("PageRank vs Citation Count\n(80k sample, colored by cluster)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=6, markerscale=4, ncol=3, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/figures/network/pagerank_vs_citations.png", dpi=150, bbox_inches="tight")
plt.close()
del scatter
gc.collect()
print("Saved pagerank_vs_citations.png")

#Figure 3: Cross-cluster heatmap 
print(f"\nBuilding cross-cluster heatmap …")
k        = n_clusters
c_to_row = {c: i for i, c in enumerate(unique_clusters)}
cross    = np.zeros((k, k), dtype=np.int64)
sampled  = 0

pf2 = pq.ParquetFile(LOCAL_PARQUET)
for g in range(n_groups):
    if sampled >= CROSS_CLUSTER_SAMPLE:
        break
    rg = pf2.read_row_group(g, columns=["id", "references"]).to_pandas()
    rg["references"] = rg["references"].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )
    rg = rg[rg["references"].map(len) > 0]
    if len(rg) == 0:
        del rg; continue

    exploded = rg.explode("references").dropna(subset=["references"])
    for _, row in exploded.iterrows():
        if sampled >= CROSS_CLUSTER_SAMPLE:
            break
        cc = id_to_cluster.get(row["id"], -1)
        cd = id_to_cluster.get(row["references"], -1)
        if cc >= 0 and cd >= 0 and cc in c_to_row and cd in c_to_row:
            cross[c_to_row[cc], c_to_row[cd]] += 1
            sampled += 1
    del rg, exploded
    gc.collect()

print(f"  Sampled {sampled:,} edges")
row_sums = cross.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cross_norm = cross / row_sums

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cross_norm, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Proportion of outgoing citations")
ax.set_xticks(range(k)); ax.set_xticklabels([f"C{c}" for c in unique_clusters], fontsize=8)
ax.set_yticks(range(k)); ax.set_yticklabels([f"C{c}" for c in unique_clusters], fontsize=8)
ax.set_xlabel("Cited cluster", fontsize=11)
ax.set_ylabel("Citing cluster", fontsize=11)
ax.set_title("Cross-Cluster Citation Heatmap\n(row = citing, col = cited; normalised by row)",
             fontsize=13, fontweight="bold")
for i in range(k):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="steelblue", linewidth=2))
plt.tight_layout()
plt.savefig("outputs/figures/network/cross_cluster_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved cross_cluster_heatmap.png")
for i, c in enumerate(unique_clusters):
    print(f"  Cluster {c:2d} self-citation: {cross_norm[i, i]:.2%}")

# Figure 4: PageRank per cluster box plot 
labelled      = meta[(meta["cluster"] >= 0) & meta["pagerank"].notna()]
cluster_order = (labelled.groupby("cluster")["pagerank"]
                 .median().sort_values(ascending=False).index.tolist())
data_per_c    = [labelled[labelled["cluster"] == c]["pagerank"].values for c in cluster_order]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data_per_c, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))
for patch, c in zip(bp["boxes"], cluster_order):
    patch.set_facecolor(cluster_color(c))
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(cluster_order) + 1))
ax.set_xticklabels([f"C{c}" for c in cluster_order], fontsize=9)
ax.set_ylabel("PageRank Score", fontsize=11)
ax.set_title("PageRank Distribution per Cluster\n(sorted by median, outliers hidden)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/network/pagerank_per_cluster_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pagerank_per_cluster_boxplot.png")

# Figure 5: Citation volume by cluster over time 
meta["decade"] = (meta["year"] // 10 * 10).astype("Int64")
timeline = (meta[(meta["cluster"] >= 0) & meta["decade"].notna()]
            .groupby(["decade", "cluster"])["in_degree"]
            .sum().unstack(fill_value=0))
timeline = timeline.loc[(timeline.index >= 1980) & (timeline.index <= 2020)]

fig, ax = plt.subplots(figsize=(13, 6))
for c in timeline.columns:
    ax.plot(timeline.index, timeline[c], marker="o", linewidth=1.8,
            color=cluster_color(c), label=f"C{c}")
ax.set_xlabel("Decade", fontsize=11)
ax.set_ylabel("Total In-Network Citations Received", fontsize=11)
ax.set_title("Citation Volume by Cluster Over Time", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/figures/network/cluster_citation_volume_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved cluster_citation_volume_over_time.png")

# Figure 6: Subgraph of top-N papers 
print(f"\nRendering subgraph of top {SUBGRAPH_TOP_N} papers …")
pr_sub      = {n: pagerank[n] for n in SG.nodes()}
cluster_sub = {n: id_to_cluster.get(n, -1) for n in SG.nodes()}
node_colors = [cluster_color(cluster_sub[n]) for n in SG.nodes()]
node_sizes  = [max(20, pr_sub[n] * 5e5) for n in SG.nodes()]

pos = nx.spring_layout(SG, seed=42, k=0.4)

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_edges(SG, pos, ax=ax, alpha=0.15, edge_color="gray",
                       arrows=True, arrowsize=6, width=0.5)
nx.draw_networkx_nodes(SG, pos, ax=ax,
                       node_color=node_colors, node_size=node_sizes, alpha=0.85)
present = sorted(set(cluster_sub.values()))
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=cluster_color(c), markersize=9,
                      label=f"Cluster {c}" if c >= 0 else "Unlabelled")
           for c in present]
ax.legend(handles=handles, fontsize=8, loc="lower left", ncol=2)
ax.set_title(f"Citation Subgraph — Top {SUBGRAPH_TOP_N} Papers by PageRank\n"
             f"(node size ∝ PageRank, color = cluster)", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/network/subgraph_top_papers.png", dpi=150, bbox_inches="tight")
plt.close()
del SG
gc.collect()
print("Saved subgraph_top_papers.png")

# Summary 
print("\n" + "=" * 60)
print("NETWORK SUMMARY")
print("=" * 60)
print(f"Total papers:  {total_papers:>12,}")
print(f"Total edges:   {total_edges:>12,}")
print("\nTop 5 papers by PageRank:")
for _, r in meta.nlargest(5, "pagerank").iterrows():
    print(f"  [{int(r.year)}] C{int(r.cluster)}  PR={r.pagerank:.6f}  {str(r.title)[:70]}")
print("\nMedian PageRank by cluster (descending):")
med = meta[meta["cluster"] >= 0].groupby("cluster")["pagerank"].median().sort_values(ascending=False)
for c, v in med.items():
    print(f"  Cluster {c:2d}: {v:.8f}")
print("\nAll figures → outputs/figures/network/")
print("Metrics    → outputs/data/network_metrics.parquet")
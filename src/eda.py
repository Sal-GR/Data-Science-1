import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import boto3
import os
import gc

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#2563EB" 
PALETTE = "Blues_d"

print("Downloading cleaned dataset from S3...")
s3 = boto3.client("s3")
os.makedirs("cleaned", exist_ok=True)
s3.download_file(
    "data-science-citation-network",
    "cleaned/papers.parquet",
    "cleaned/papers.parquet"
)
COLS = ["n_citation", "year", "venue", "author_count", "reference_count",
        "abstract", "authors", "references"]
df = pd.read_parquet("cleaned/papers.parquet", columns=COLS)
print(f"Loaded {len(df):,} papers.")

os.makedirs("outputs/figures/eda", exist_ok=True)

# A1:  Missing value heatmap summary bar chart
print("Plotting missing data summary...")

missing_counts = {
    "abstract": int((df["abstract"] == "").sum()),
    "venue":    int((df["venue"] == "").sum()),
    "authors":  int(df["authors"].apply(lambda x: not isinstance(x, list) or len(x) == 0).sum()),
    "references": int(df["references"].apply(lambda x: not isinstance(x, list) or len(x) == 0).sum()),
}
missing_pct = {k: v / len(df) * 100 for k, v in missing_counts.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar: absolute counts
bars = axes[0].barh(list(missing_counts.keys()), list(missing_counts.values()), color=ACCENT, edgecolor="white")
axes[0].set_xlabel("Number of papers")
axes[0].set_title("Missing / Empty Fields — Absolute Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, missing_counts.values()):
    axes[0].text(bar.get_width() + len(df) * 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=10)

# Bar: percentages
bars2 = axes[1].barh(list(missing_pct.keys()), list(missing_pct.values()), color="#64748B", edgecolor="white")
axes[1].set_xlabel("% of total papers")
axes[1].set_title("Missing / Empty Fields — Percentage")
for bar, val in zip(bars2, missing_pct.values()):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10)

fig.suptitle("Data Completeness Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/eda/missing_data_summary.png", dpi=150, bbox_inches="tight")
plt.close()
gc.collect()

print("Plotting citation count distribution...")

# B1:  Full distribution (log-scaled x-axis) — reveals right skew clearly
fig, ax = plt.subplots(figsize=(10, 5))
nonzero = df[df["n_citation"] > 0]["n_citation"]
ax.hist(nonzero, bins=100, color=ACCENT, edgecolor="white", linewidth=0.4)
ax.set_xscale("log")
ax.set_xlabel("Citation count (log scale)")
ax.set_ylabel("Number of papers")
ax.set_title("Distribution of Citation Counts (papers with ≥1 citation, log scale)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Annotate median and 95th percentile
median_c = nonzero.median()
p95_c    = nonzero.quantile(0.95)
for val, label, color in [(median_c, f"Median\n{median_c:.0f}", "#16A34A"),
                           (p95_c,   f"95th pct\n{p95_c:.0f}", "#DC2626")]:
    ax.axvline(val, color=color, linestyle="--", linewidth=1.5)
    ax.text(val * 1.1, ax.get_ylim()[1] * 0.85, label, color=color, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/eda/citation_distribution_log.png", dpi=150, bbox_inches="tight")
plt.close()

# B2:  Zoomed-in view: papers with 0–50 citations (most papers live here)
fig, ax = plt.subplots(figsize=(10, 5))
low_cite = df[df["n_citation"] <= 50]["n_citation"]
ax.hist(low_cite, bins=51, color=ACCENT, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Citation count")
ax.set_ylabel("Number of papers")
ax.set_title("Citation Count Distribution — Zoomed: 0 to 50 Citations")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/figures/eda/citation_distribution_zoomed.png", dpi=150, bbox_inches="tight")
plt.close()

# B3:  Boxplot by era (pre-2000, 2000-2009, 2010+)
print("Plotting citation boxplot by era...")
df["era"] = pd.cut(df["year"],
                   bins=[0, 1999, 2009, 9999],
                   labels=["Pre-2000", "2000–2009", "2010+"])
era_data = df[df["n_citation"] > 0]

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=era_data, x="era", y="n_citation", hue="era", palette="Blues",
            showfliers=False, legend=False, ax=ax)
ax.set_yscale("log")
ax.set_xlabel("Publication era")
ax.set_ylabel("Citation count (log scale)")
ax.set_title("Citation Count by Era (outliers hidden, log scale)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/citation_boxplot_by_era.png", dpi=150, bbox_inches="tight")
plt.close()

# B4:  CDF of citation counts — sample 100k points to avoid OOM on runner
fig, ax = plt.subplots(figsize=(10, 5))
sample_size = min(100_000, len(df))
cdf_sample = df["n_citation"].sample(sample_size, random_state=42)
sorted_citations = np.sort(cdf_sample.values)
cdf = np.arange(1, len(sorted_citations) + 1) / len(sorted_citations)
ax.plot(sorted_citations, cdf, color=ACCENT, linewidth=1.5)
ax.set_xscale("log")
ax.set_xlabel("Citation count (log scale)")
ax.set_ylabel("Cumulative proportion of papers")
ax.set_title("Cumulative Distribution of Citation Counts")

# Annotate: what % of papers have < 10 citations?
threshold = 10
pct_below = (df["n_citation"] < threshold).mean() * 100
ax.axvline(threshold, color="#DC2626", linestyle="--", linewidth=1.2)
ax.text(threshold * 1.2, 0.3, f"{pct_below:.1f}% of papers\nhave < {threshold} citations",
        color="#DC2626", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/eda/citation_cdf.png", dpi=150, bbox_inches="tight")
plt.close()
del nonzero, low_cite, era_data, sorted_citations, cdf
gc.collect()

print("Plotting papers per year...")

year_counts = df.groupby("year").size().reset_index(name="count")
year_counts = year_counts[(year_counts["year"] >= 1950) & (year_counts["year"] <= 2020)]

# C1:  Bar chart: papers per year
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(year_counts["year"], year_counts["count"], color=ACCENT, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Year")
ax.set_ylabel("Number of papers")
ax.set_title("Papers Published Per Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/figures/eda/papers_per_year.png", dpi=150, bbox_inches="tight")
plt.close()

# C2:  Cumulative papers over time
fig, ax = plt.subplots(figsize=(14, 5))
year_counts["cumulative"] = year_counts["count"].cumsum()
ax.fill_between(year_counts["year"], year_counts["cumulative"], alpha=0.3, color=ACCENT)
ax.plot(year_counts["year"], year_counts["cumulative"], color=ACCENT, linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative papers")
ax.set_title("Cumulative Paper Count Over Time")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/figures/eda/papers_cumulative.png", dpi=150, bbox_inches="tight")
plt.close()

# C3:  Median citations per year (shows how older papers accumulate more)
print("Plotting median citations per year...")
cite_by_year = df[(df["year"] >= 1950) & (df["year"] <= 2020)].groupby("year")["n_citation"].median().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(cite_by_year["year"], cite_by_year["n_citation"], color=ACCENT, linewidth=2, marker="o", markersize=3)
ax.set_xlabel("Year")
ax.set_ylabel("Median citation count")
ax.set_title("Median Citation Count by Publication Year\n(older papers have had more time to accumulate citations)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/median_citations_by_year.png", dpi=150, bbox_inches="tight")
plt.close()
del year_counts, cite_by_year
gc.collect()

print("Plotting venue analysis...")

venue_df = df[df["venue"].str.strip() != ""]

# D1:  Top 20 venues by paper count
top_venues = venue_df["venue"].value_counts().head(20).reset_index()
top_venues.columns = ["venue", "count"]

fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=top_venues, y="venue", x="count", hue="venue", palette=PALETTE, legend=False, ax=ax)
ax.set_xlabel("Number of papers")
ax.set_ylabel("")
ax.set_title("Top 20 Venues by Paper Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/figures/eda/top_venues_by_count.png", dpi=150, bbox_inches="tight")
plt.close()

# D2:  Top 20 venues by median citation count
venue_cite = (venue_df.groupby("venue")["n_citation"]
              .agg(median="median", count="count")
              .reset_index())
venue_cite = venue_cite[venue_cite["count"] >= 50]
top_cite_venues = venue_cite.nlargest(20, "median")

fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=top_cite_venues, y="venue", x="median", hue="venue", palette=PALETTE, legend=False, ax=ax)
ax.set_xlabel("Median citation count")
ax.set_ylabel("")
ax.set_title("Top 20 Venues by Median Citation Count\n(venues with ≥ 50 papers)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/top_venues_by_citations.png", dpi=150, bbox_inches="tight")
plt.close()

# D3:  Papers per year for top 5 venues (trend lines)
print("Plotting venue trends over time...")
top5_venues = top_venues["venue"].head(5).tolist()
venue_year = (venue_df[venue_df["venue"].isin(top5_venues)]
              .groupby(["venue", "year"])
              .size()
              .reset_index(name="count"))
venue_year = venue_year[(venue_year["year"] >= 1990) & (venue_year["year"] <= 2020)]

fig, ax = plt.subplots(figsize=(13, 6))
for venue in top5_venues:
    sub = venue_year[venue_year["venue"] == venue]
    ax.plot(sub["year"], sub["count"], marker="o", markersize=3, linewidth=1.8, label=venue)
ax.set_xlabel("Year")
ax.set_ylabel("Papers published")
ax.set_title("Paper Output Over Time — Top 5 Venues")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/figures/eda/top5_venues_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
del venue_df, top_venues, venue_cite, top_cite_venues, venue_year
gc.collect()

print("Plotting author count distribution...")

# E1:  Distribution of authors per paper (cap at 15 for readability)
author_capped = df["author_count"].clip(upper=15)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(author_capped, bins=range(0, 17), color=ACCENT, edgecolor="white", linewidth=0.6, align="left")
ax.set_xlabel("Number of authors per paper")
ax.set_ylabel("Number of papers")
ax.set_title("Distribution of Authors per Paper (capped at 15)")
ax.set_xticks(range(0, 16))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Annotate mean
mean_auth = df["author_count"].mean()
ax.axvline(mean_auth, color="#DC2626", linestyle="--", linewidth=1.5)
ax.text(mean_auth + 0.1, ax.get_ylim()[1] * 0.88, f"Mean: {mean_auth:.1f}", color="#DC2626", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/eda/author_count_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# E2:  Average authors per paper over time (collaboration trends)
print("Plotting collaboration trends over time...")
collab = df[(df["year"] >= 1950) & (df["year"] <= 2020)].groupby("year")["author_count"].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(collab["year"], collab["author_count"], color=ACCENT, linewidth=2, marker="o", markersize=3)
ax.set_xlabel("Year")
ax.set_ylabel("Average authors per paper")
ax.set_title("Average Number of Authors per Paper Over Time\n(tracks growth of collaborative research)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/avg_authors_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# E3:  Do papers with more authors get more citations?
print("Plotting author count vs citations...")
author_bucket_col = pd.cut(df["author_count"],
                           bins=[0, 1, 2, 3, 5, 10, 9999],
                           labels=["1", "2", "3", "4–5", "6–10", "11+"])
cite_by_auth = (df.assign(author_bucket=author_bucket_col)
                  .groupby("author_bucket", observed=True)["n_citation"]
                  .median()
                  .reset_index())

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=cite_by_auth, x="author_bucket", y="n_citation", hue="author_bucket", palette=PALETTE, legend=False, ax=ax)
ax.set_xlabel("Number of authors")
ax.set_ylabel("Median citation count")
ax.set_title("Median Citations by Author Count\n(do larger teams produce more-cited work?)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/citations_by_author_count.png", dpi=150, bbox_inches="tight")
plt.close()
del author_capped, collab, cite_by_auth, author_bucket_col
gc.collect()

print("Plotting reference count distribution...")

# F1:  Distribution of references per paper
ref_capped = df["reference_count"].clip(upper=100)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ref_capped, bins=50, color="#64748B", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Number of references per paper (capped at 100)")
ax.set_ylabel("Number of papers")
ax.set_title("Distribution of Reference Counts per Paper")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
mean_ref = df["reference_count"].mean()
ax.axvline(mean_ref, color="#DC2626", linestyle="--", linewidth=1.5)
ax.text(mean_ref + 0.5, ax.get_ylim()[1] * 0.88, f"Mean: {mean_ref:.1f}", color="#DC2626", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/eda/reference_count_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n════════ EDA Summary Statistics ════════")
print(f"Total papers:              {len(df):,}")
print(f"Year range:                {df['year'].min()} – {df['year'].max()}")
print(f"Unique venues:             {df['venue'].nunique():,}")
print(f"Papers with no abstract:   {(df['abstract'] == '').sum():,} ({(df['abstract'] == '').mean()*100:.1f}%)")
print(f"Papers with no venue:      {(df['venue'] == '').sum():,} ({(df['venue'] == '').mean()*100:.1f}%)")
print(f"Avg citation count:        {df['n_citation'].mean():.2f}")
print(f"Median citation count:     {df['n_citation'].median():.0f}")
print(f"Max citation count:        {df['n_citation'].max():,}")
print(f"Papers with 0 citations:   {(df['n_citation'] == 0).sum():,} ({(df['n_citation'] == 0).mean()*100:.1f}%)")
print(f"Avg authors per paper:     {df['author_count'].mean():.2f}")
print(f"Avg references per paper:  {df['reference_count'].mean():.2f}")
print("════════════════════════════════════════\n")

print("EDA complete. All figures saved to outputs/figures/eda/")
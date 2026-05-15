"""
RAG System — Full EDA Dashboard
================================
Run:  python rag_eda.py
Or:   paste each section into a Jupyter notebook cell-by-cell.

Outputs: a folder called  rag_eda_plots/  with one PNG per chart.

Required packages:
    pip install pandas matplotlib seaborn scikit-learn umap-learn nltk
Optional (richer topic chart):
    pip install wordcloud
"""

# ── 0. IMPORTS & SETUP ────────────────────────────────────────────────────────

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter

warnings.filterwarnings("ignore")
os.makedirs("rag_eda_plots", exist_ok=True)

# Consistent style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

ACCENT   = "#4F7CFF"   # blue
WARN     = "#F59E0B"   # amber
SUCCESS  = "#10B981"   # green
NEUTRAL  = "#94A3B8"   # slate


# ── 1. SAMPLE DATA GENERATION ─────────────────────────────────────────────────
#
#  REPLACE THIS SECTION with your real data loading, e.g.:
#
#      query_df     = pd.read_csv("query_logs.csv",     parse_dates=["timestamp"])
#      retrieval_df = pd.read_csv("retrieval_logs.csv")
#      corpus_df    = pd.read_csv("doc_corpus.csv",     parse_dates=["created_date"])
#      feedback_df  = pd.read_csv("feedback_logs.csv")

rng = np.random.default_rng(42)
N   = 2000   # number of query events

# ── query_logs ────────────────────────────────────────────────────────────────
roles       = rng.choice(["engineer", "analyst", "manager", "hr", "legal"],
                          N, p=[0.30, 0.25, 0.20, 0.15, 0.10])
departments = rng.choice(["Engineering", "Finance", "HR", "Legal", "Product"],
                          N, p=[0.28, 0.22, 0.18, 0.12, 0.20])
topics      = rng.choice(
    ["onboarding", "leave policy", "deployment", "compliance",
     "expense report", "security", "performance review", "API docs",
     "benefits", "data retention"],
    N, p=[0.18, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
)

base_ts = datetime(2024, 6, 1)
timestamps = [base_ts + timedelta(
    days=int(rng.integers(0, 180)),
    hours=int(rng.integers(7, 22)),
    minutes=int(rng.integers(0, 60))
) for _ in range(N)]

query_texts = [f"What is the {t} process?" if rng.random() > 0.4
               else f"How do I submit a {t} request?" for t in topics]

query_df = pd.DataFrame({
    "query_id"   : [f"q{i:04d}" for i in range(N)],
    "user_id"    : [f"u{rng.integers(1, 200):03d}" for _ in range(N)],
    "role"       : roles,
    "department" : departments,
    "topic"      : topics,
    "query_text" : query_texts,
    "timestamp"  : timestamps,
    "latency_ms" : np.clip(rng.lognormal(6.0, 0.7, N), 200, 8000).astype(int),
    "search_type": rng.choice(["semantic", "keyword", "hybrid"], N, p=[0.45, 0.25, 0.30]),
})
query_df["word_count"] = query_df["query_text"].str.split().str.len()

# ── retrieval_logs ─────────────────────────────────────────────────────────────
R = N * 5   # 5 retrieved chunks per query on average
retrieval_df = pd.DataFrame({
    "query_id"        : rng.choice(query_df["query_id"], R),
    "chunk_id"        : [f"c{rng.integers(0, 400):04d}" for _ in range(R)],
    "rank"            : rng.integers(1, 6, R),
    "similarity_score": np.clip(rng.beta(5, 2, R), 0.3, 1.0).round(3),
    "search_type"     : rng.choice(["semantic", "keyword", "hybrid"], R, p=[0.45, 0.25, 0.30]),
})

# ── doc_corpus ─────────────────────────────────────────────────────────────────
C = 400
corpus_df = pd.DataFrame({
    "chunk_id"    : [f"c{i:04d}" for i in range(C)],
    "source_file" : [f"doc_{rng.integers(1, 80):03d}" for _ in range(C)],
    "file_type"   : rng.choice(["pdf", "docx", "email"], C, p=[0.50, 0.30, 0.20]),
    "department"  : rng.choice(["Engineering", "Finance", "HR", "Legal", "Product"], C),
    "created_date": [datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1600)))
                     for _ in range(C)],
    "token_count" : rng.integers(50, 800, C),
})

# ── feedback_logs ──────────────────────────────────────────────────────────────
sampled_ids = rng.choice(query_df["query_id"], int(N * 0.6), replace=False)
feedback_df = pd.DataFrame({
    "query_id"       : sampled_ids,
    "thumbs_up"      : rng.choice([True, False], len(sampled_ids), p=[0.72, 0.28]),
    "relevance_score": np.clip(rng.normal(3.8, 0.9, len(sampled_ids)), 1, 5).round(1),
    "citation_used"  : rng.choice([True, False], len(sampled_ids), p=[0.65, 0.35]),
})


# ── HELPER ─────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = f"rag_eda_plots/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  saved → {path}")


print("\n📊  RAG EDA — generating plots …\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — QUERY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── A1. Query volume over time (daily) ────────────────────────────────────────
query_df["date"] = pd.to_datetime(query_df["timestamp"]).dt.date
daily = query_df.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(daily["date"], daily["count"], alpha=0.15, color=ACCENT)
ax.plot(daily["date"], daily["count"], color=ACCENT, lw=1.8)
ax.set_title("A1 · Query volume over time")
ax.set_xlabel("Date"); ax.set_ylabel("Queries / day")
# 7-day rolling average
daily["roll7"] = daily["count"].rolling(7, center=True).mean()
ax.plot(daily["date"], daily["roll7"], color=WARN, lw=2, linestyle="--", label="7-day avg")
ax.legend(frameon=False)
save(fig, "A1_query_volume_over_time")

# ── A2. Heatmap — queries by hour × weekday ───────────────────────────────────
query_df["hour"]    = pd.to_datetime(query_df["timestamp"]).dt.hour
query_df["weekday"] = pd.to_datetime(query_df["timestamp"]).dt.day_name()
heat = (query_df.groupby(["weekday", "hour"])
                .size()
                .unstack(fill_value=0)
                .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(heat, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="#f0f0f0",
            cbar_kws={"label": "query count"})
ax.set_title("A2 · Query heatmap — hour of day × weekday")
ax.set_xlabel("Hour of day"); ax.set_ylabel("")
save(fig, "A2_query_heatmap_hour_weekday")

# ── A3. Query length distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(query_df["word_count"], bins=20, color=ACCENT, edgecolor="white", alpha=0.85)
med = query_df["word_count"].median()
ax.axvline(med, color=WARN, lw=2, linestyle="--", label=f"Median = {med:.0f} words")
ax.set_title("A3 · Query length distribution")
ax.set_xlabel("Word count"); ax.set_ylabel("Number of queries")
ax.legend(frameon=False)
save(fig, "A3_query_length_distribution")

# ── A4. Top query topics ───────────────────────────────────────────────────────
topic_counts = query_df["topic"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(topic_counts.index[::-1], topic_counts.values[::-1],
               color=ACCENT, edgecolor="white")
for bar, val in zip(bars, topic_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, color=NEUTRAL)
ax.set_title("A4 · Top query topics")
ax.set_xlabel("Query count")
save(fig, "A4_top_query_topics")

# ── A5. Queries by role ────────────────────────────────────────────────────────
role_counts = query_df["role"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
colors = sns.color_palette("muted", len(role_counts))
ax.bar(role_counts.index, role_counts.values, color=colors, edgecolor="white")
ax.set_title("A5 · Query volume by user role")
ax.set_ylabel("Query count")
save(fig, "A5_queries_by_role")

# ── A6. Latency distribution (log scale) ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(query_df["latency_ms"], bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
for pct, col in [(50, SUCCESS), (95, WARN), (99, "red")]:
    v = np.percentile(query_df["latency_ms"], pct)
    axes[0].axvline(v, color=col, lw=1.8, linestyle="--", label=f"P{pct} {v:.0f}ms")
axes[0].set_title("A6a · Latency distribution")
axes[0].set_xlabel("Latency (ms)"); axes[0].set_ylabel("Count")
axes[0].legend(frameon=False, fontsize=9)

# Latency by search type
search_lat = query_df.groupby("search_type")["latency_ms"].agg(
    P50=("median"),
    P95=(lambda x: x.quantile(0.95))
).rename(columns={"P50":"P50","P95":"P95"})
x = np.arange(len(search_lat))
w = 0.35
axes[1].bar(x - w/2, search_lat["P50"], w, label="P50", color=SUCCESS, edgecolor="white")
axes[1].bar(x + w/2, search_lat["P95"], w, label="P95", color=WARN,    edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(search_lat.index)
axes[1].set_title("A6b · Latency by search type")
axes[1].set_ylabel("Latency (ms)")
axes[1].legend(frameon=False)

fig.tight_layout()
save(fig, "A6_latency_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — RETRIEVAL QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

# ── B1. Similarity score distribution by search type ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for stype, color in zip(["semantic","hybrid","keyword"], [ACCENT, SUCCESS, WARN]):
    data = retrieval_df[retrieval_df["search_type"] == stype]["similarity_score"]
    ax.hist(data, bins=30, alpha=0.55, label=stype, color=color, edgecolor="white")
ax.axvline(0.6, color="red", lw=1.5, linestyle="--", label="Low-match threshold (0.6)")
ax.set_title("B1 · Similarity score distribution by search type")
ax.set_xlabel("Cosine similarity score")
ax.set_ylabel("Count")
ax.legend(frameon=False)
save(fig, "B1_similarity_score_distribution")

# ── B2. Chunk retrieval frequency (long-tail) ─────────────────────────────────
chunk_freq = retrieval_df["chunk_id"].value_counts().reset_index()
chunk_freq.columns = ["chunk_id", "retrieval_count"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Top 20 most retrieved chunks
top20 = chunk_freq.head(20)
axes[0].barh(top20["chunk_id"][::-1], top20["retrieval_count"][::-1],
             color=ACCENT, edgecolor="white")
axes[0].set_title("B2a · Top 20 most-retrieved chunks")
axes[0].set_xlabel("Retrieval count")

# Full frequency histogram (long tail)
axes[1].hist(chunk_freq["retrieval_count"], bins=30, color=NEUTRAL, edgecolor="white")
never_retrieved = len(corpus_df) - len(chunk_freq)
axes[1].set_title(f"B2b · Chunk retrieval frequency\n({never_retrieved} chunks never retrieved)")
axes[1].set_xlabel("Times retrieved")
axes[1].set_ylabel("Number of chunks")

fig.tight_layout()
save(fig, "B2_chunk_retrieval_frequency")

# ── B3. Average similarity score by search type ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
avg_sim = (retrieval_df.groupby("search_type")["similarity_score"]
                       .agg(["mean","std"])
                       .reset_index())
colors = [ACCENT, SUCCESS, WARN]
bars = ax.bar(avg_sim["search_type"], avg_sim["mean"],
              yerr=avg_sim["std"], capsize=5,
              color=colors, edgecolor="white", alpha=0.85)
ax.set_title("B3 · Avg similarity score by search type (±1 std)")
ax.set_ylabel("Mean cosine similarity")
ax.set_ylim(0.5, 1.0)
for bar, val in zip(bars, avg_sim["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", fontsize=10)
save(fig, "B3_avg_similarity_by_search_type")

# ── B4. Rank position vs user feedback ────────────────────────────────────────
ret_fb = retrieval_df.merge(
    query_df[["query_id"]].merge(feedback_df, on="query_id"), on="query_id"
)
rank_feedback = (ret_fb.groupby("rank")["thumbs_up"]
                        .apply(lambda x: x.map({True:1,False:0}).mean())
                        .reset_index())

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rank_feedback["rank"], rank_feedback["thumbs_up"],
        marker="o", color=ACCENT, lw=2.5, ms=8)
ax.fill_between(rank_feedback["rank"], rank_feedback["thumbs_up"], alpha=0.1, color=ACCENT)
ax.set_title("B4 · Thumbs-up rate by retrieval rank position")
ax.set_xlabel("Rank (1 = top retrieved chunk)")
ax.set_ylabel("Thumbs-up rate")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
save(fig, "B4_rank_vs_feedback")

# ── B5. Citation usage vs thumbs-up ───────────────────────────────────────────
cit_fb = feedback_df.groupby("citation_used")["thumbs_up"].apply(
    lambda x: x.map({True:1,False:0}).mean()
).reset_index()
cit_fb["label"] = cit_fb["citation_used"].map({True:"With citation", False:"No citation"})

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(cit_fb["label"], cit_fb["thumbs_up"],
       color=[SUCCESS, NEUTRAL], edgecolor="white", width=0.5)
for i, val in enumerate(cit_fb["thumbs_up"]):
    ax.text(i, val + 0.01, f"{val:.1%}", ha="center", fontsize=11)
ax.set_title("B5 · Thumbs-up rate: cited vs uncited answers")
ax.set_ylabel("Thumbs-up rate")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.05)
save(fig, "B5_citation_vs_feedback")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — KNOWLEDGE BASE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

# ── C1. Corpus composition — file type × department ───────────────────────────
comp = corpus_df.groupby(["department","file_type"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(9, 5))
comp.plot(kind="bar", ax=ax, edgecolor="white", colormap="Set2")
ax.set_title("C1 · Corpus composition by department and file type")
ax.set_xlabel("Department"); ax.set_ylabel("Number of chunks")
ax.tick_params(axis="x", rotation=25)
ax.legend(title="File type", frameon=False)
save(fig, "C1_corpus_composition")

# ── C2. Document staleness — retrieval rate by doc age ────────────────────────
corpus_df["age_days"] = (datetime(2024, 6, 1) - corpus_df["created_date"]).dt.days
corpus_df["age_bucket"] = pd.cut(
    corpus_df["age_days"],
    bins=[0, 180, 365, 730, 1095, 9999],
    labels=["<6 mo","6–12 mo","1–2 yr","2–3 yr","3+ yr"]
)
chunk_hits = set(retrieval_df["chunk_id"].unique())
corpus_df["retrieved"] = corpus_df["chunk_id"].isin(chunk_hits)
stale = (corpus_df.groupby("age_bucket", observed=True)["retrieved"]
                   .mean()
                   .reset_index())

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = [SUCCESS if v > 0.5 else WARN if v > 0.3 else "tomato"
              for v in stale["retrieved"]]
ax.bar(stale["age_bucket"].astype(str), stale["retrieved"],
       color=bar_colors, edgecolor="white")
for i, val in enumerate(stale["retrieved"]):
    ax.text(i, val + 0.01, f"{val:.0%}", ha="center", fontsize=10)
ax.set_title("C2 · Retrieval rate by document age\n(low % = stale / outdated content)")
ax.set_xlabel("Document age"); ax.set_ylabel("% chunks retrieved at least once")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
save(fig, "C2_document_staleness")

# ── C3. Role × topic heatmap (access pattern) ─────────────────────────────────
role_topic = (query_df.groupby(["role","topic"])
                       .size()
                       .unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(role_topic, ax=ax, cmap="Blues", linewidths=0.3,
            linecolor="#e8e8e8", annot=True, fmt="d", annot_kws={"size": 8})
ax.set_title("C3 · Role × topic access heatmap\n(cross-role spikes may indicate access-control gaps)")
ax.set_xlabel("Topic"); ax.set_ylabel("Role")
ax.tick_params(axis="x", rotation=30)
save(fig, "C3_role_topic_heatmap")

# ── C4. Knowledge gap map — UMAP / t-SNE of query embeddings ──────────────────
#
# This uses TF-IDF + PCA as a lightweight proxy for real embeddings.
# To use real embeddings: replace tfidf_matrix with your actual embedding matrix
# (shape: N_queries × embedding_dim), then run UMAP on it.
#
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    vec = TfidfVectorizer(max_features=200, stop_words="english")
    X   = vec.fit_transform(query_df["query_text"]).toarray()

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        coords  = reducer.fit_transform(X)
        method_label = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
        method_label = "t-SNE"

    plot_df = query_df.copy()
    plot_df["x"], plot_df["y"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab10", n_colors=query_df["topic"].nunique())
    for i, (topic, grp) in enumerate(plot_df.groupby("topic")):
        ax.scatter(grp["x"], grp["y"], label=topic, alpha=0.55,
                   s=25, color=palette[i], edgecolors="none")
    ax.set_title(f"C4 · Query embedding map ({method_label})\n"
                 "Clusters = related intent groups. Isolated points = niche or unclear queries.")
    ax.set_xlabel(f"{method_label} dim 1")
    ax.set_ylabel(f"{method_label} dim 2")
    ax.legend(title="Topic", bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, fontsize=9)
    save(fig, "C4_knowledge_gap_map")

except Exception as e:
    print(f"  ⚠  C4 skipped — {e}")

# ── C5. Token count distribution across corpus ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(corpus_df["token_count"], bins=30, color=SUCCESS, edgecolor="white", alpha=0.85)
ax.axvline(corpus_df["token_count"].median(), color=WARN, lw=2,
           linestyle="--", label=f"Median = {corpus_df['token_count'].median():.0f} tokens")
ax.set_title("C5 · Chunk token count distribution\n"
             "(very short chunks = low context; very long = may dilute relevance)")
ax.set_xlabel("Tokens per chunk"); ax.set_ylabel("Number of chunks")
ax.legend(frameon=False)
save(fig, "C5_chunk_token_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════╗
║            RAG EDA — all plots saved                     ║
╠══════════════════════════════════════════════════════════╣
║  rag_eda_plots/                                          ║
║  ├── A1_query_volume_over_time.png                       ║
║  ├── A2_query_heatmap_hour_weekday.png                   ║
║  ├── A3_query_length_distribution.png                    ║
║  ├── A4_top_query_topics.png                             ║
║  ├── A5_queries_by_role.png                              ║
║  ├── A6_latency_distribution.png                         ║
║  ├── B1_similarity_score_distribution.png                ║
║  ├── B2_chunk_retrieval_frequency.png                    ║
║  ├── B3_avg_similarity_by_search_type.png                ║
║  ├── B4_rank_vs_feedback.png                             ║
║  ├── B5_citation_vs_feedback.png                         ║
║  ├── C1_corpus_composition.png                           ║
║  ├── C2_document_staleness.png                           ║
║  ├── C3_role_topic_heatmap.png                           ║
║  ├── C4_knowledge_gap_map.png                            ║
║  └── C5_chunk_token_distribution.png                     ║
╚══════════════════════════════════════════════════════════╝

Next steps:
  → Replace the "SAMPLE DATA" section with pd.read_csv() calls
  → For C4 with real embeddings: swap tfidf_matrix for your
    embedding matrix (query_df shape: N x embedding_dim)
  → pip install umap-learn  for better gap-map quality
""")

# 📊 RAG EDA Dashboard
> Exploratory Data Analysis for an Enterprise Knowledge Assistant (RAG System)

---

## 🧠 What is this project?

This project performs a full **Exploratory Data Analysis (EDA)** on an enterprise **Retrieval-Augmented Generation (RAG)** system — a ChatGPT-like assistant trained on company documents (PDFs, Word files, emails).

It generates **16 diagnostic plots** across 3 sections to help understand system performance, retrieval quality, and knowledge base health.

---

## 🏗️ System Architecture

```
User Query
    ↓
Hybrid Search (Semantic + Keyword)
    ↓
Vector Database → Retrieved Chunks → Citations
    ↓
LLM (with Role-Based Access Control)
    ↓
Answer with source citations
```

---

## 📁 Project Structure

```
rag-eda-dashboard/
├── rag_eda.py              ← main EDA script
└── rag_eda_plots/
    ├── A1_query_volume_over_time.png
    ├── A2_query_heatmap_hour_weekday.png
    ├── A3_query_length_distribution.png
    ├── A4_top_query_topics.png
    ├── A5_queries_by_role.png
    ├── A6_latency_distribution.png
    ├── B1_similarity_score_distribution.png
    ├── B2_chunk_retrieval_frequency.png
    ├── B3_avg_similarity_by_search_type.png
    ├── B4_rank_vs_feedback.png
    ├── B5_citation_vs_feedback.png
    ├── C1_corpus_composition.png
    ├── C2_document_staleness.png
    ├── C3_role_topic_heatmap.png
    ├── C4_knowledge_gap_map.png
    └── C5_chunk_token_distribution.png
```

---

## 📈 What the plots cover

### Section A — Query Analysis
| Plot | What it shows |
|------|--------------|
| A1 · Query volume over time | Daily usage trends with 7-day rolling average |
| A2 · Hour × weekday heatmap | Peak usage times across the week |
| A3 · Query length distribution | Are users asking short or conversational questions? |
| A4 · Top query topics | What employees actually need answers about |
| A5 · Queries by role | Which roles use the system most |
| A6 · Latency distribution | P50/P95/P99 response times by search type |

### Section B — Retrieval Quality
| Plot | What it shows |
|------|--------------|
| B1 · Similarity score distribution | Retrieval confidence across search modes |
| B2 · Chunk retrieval frequency | Celebrity chunks vs never-retrieved gaps |
| B3 · Avg similarity by search type | Semantic vs keyword vs hybrid comparison |
| B4 · Rank vs feedback | Does rank 1 always win thumbs-up? |
| B5 · Citation vs feedback | Do cited answers get better ratings? |

### Section C — Knowledge Base Health
| Plot | What it shows |
|------|--------------|
| C1 · Corpus composition | File types and departments in the knowledge base |
| C2 · Document staleness | Retrieval rate by document age |
| C3 · Role × topic heatmap | Access patterns — flags security gaps |
| C4 · Knowledge gap map | UMAP/t-SNE of query clusters vs corpus coverage |
| C5 · Chunk token distribution | Are chunks too short or too long? |

---

## 🚀 How to run

### 1. Install dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn umap-learn
```

### 2. Run the script
```bash
python rag_eda.py
```

### 3. View plots
All 16 plots are saved to the `rag_eda_plots/` folder.

---

## 🔌 Using your real data

Replace the sample data section in `rag_eda.py` with:

```python
query_df     = pd.read_csv("query_logs.csv",     parse_dates=["timestamp"])
retrieval_df = pd.read_csv("retrieval_logs.csv")
corpus_df    = pd.read_csv("doc_corpus.csv",     parse_dates=["created_date"])
feedback_df  = pd.read_csv("feedback_logs.csv")
```

### Expected columns

| Table | Key columns |
|-------|------------|
| query_logs | query_id, user_id, role, query_text, timestamp, latency_ms, search_type |
| retrieval_logs | query_id, chunk_id, rank, similarity_score, search_type |
| doc_corpus | chunk_id, source_file, file_type, department, created_date, token_count |
| feedback_logs | query_id, thumbs_up, relevance_score, citation_used |

---

## 🛠️ Tech stack

- **Python 3.x**
- **pandas** — data manipulation
- **matplotlib + seaborn** — visualisation
- **scikit-learn** — TF-IDF, t-SNE
- **umap-learn** — dimensionality reduction for gap map

---

## 👤 Author

**HEAVEN22-codemax**  
[github.com/HEAVEN22-codemax](https://github.com/HEAVEN22-codemax)

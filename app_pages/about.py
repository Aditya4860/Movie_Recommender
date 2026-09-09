"""About — transparent technical documentation for the CineMatch ML system."""

import streamlit as st


st.title("About CineMatch")
st.caption(
    "A transparent, content-based movie recommendation system built on the TMDB 5000 dataset."
)

# ── 1. Dataset ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/database: Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "TMDB 5000")
    col2.metric("Movies after merge", "4,800")
    col3.metric("Source", "Kaggle / TMDB API")
    st.markdown(
        """
The dataset consists of two CSV files:

| File | Contents |
|---|---|
| `tmdb_5000_movies.csv` | Title, overview, genres, keywords, ratings, runtime, popularity |
| `tmdb_5000_credits.csv` | Full cast and crew per film |

The two files are **joined on title** (inner join).  Unmatched rows are dropped,
giving a final corpus of **4,800 deduplicated movies**.
        """
    )

# ── 2. Features ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/psychology: Features")
    st.markdown(
        """
Five metadata signals are extracted per movie and combined into a single **tags** string:

| Field | Source column | Notes |
|---|---|---|
| `overview` | `overview` | Free-text plot summary |
| `genres` | `genres` (JSON list) | e.g. Action, Thriller, Science Fiction |
| `keywords` | `keywords` (JSON list) | TMDB thematic keywords |
| `cast` | `cast` (JSON list) | Top-3 billed actors only |
| `director` | `crew` (JSON list) | Filtered by `job == "Director"` |

Multi-word names are compacted to single tokens (e.g. `Christopher Nolan` → `ChristopherNolan`)
so they are not fragmented by the vectorizer.
        """
    )

# ── 3. Pipeline ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/account_tree: Pipeline")
    st.markdown(
        """
```
TMDB CSV files
    │
    ▼ src/data.py
Load & cache raw DataFrames
    │
    ▼ src/preprocessing.py  (build_feature_frame)
Merge movies + credits on title
Parse JSON-like columns (genres, keywords, cast, crew)
Extract structured tokens
Apply per-field repetition weights
Construct weighted tags string  ←  "action thriller sciencefiction ChristopherNolan inception heist …"
    │
    ▼ src/recommender.py  (MovieRecommender._fit)
Vectorize tags  →  sparse feature matrix  (5,000 features, max_features)
Compute cosine similarity  →  float32 matrix  (4,800 × 4,800)
    │
    ▼ models/  (src/model_store.py)
Persist: recommender.joblib (61.8 MB) + movies.parquet (1.8 MB)
Load once at startup  →  sub-second cold start
    │
    ▼ Query
recommend(movie_id, k)  →  top-K IDs + similarity scores  →  UI / API
```
        """
    )

# ── 4. Feature Weights ────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/tune: Feature Weighting")
    st.markdown(
        """
Each field is repeated **N times** in the tags string before vectorization,
giving it proportionally more influence in the similarity space.

| Field | Weight | Rationale |
|---|---|---|
| `overview` | **1×** | Long free text — kept at 1× to avoid drowning structured signals |
| `genres` | **3×** | Short but highly discriminative |
| `keywords` | **3×** | Thematic keywords are very distinctive |
| `cast` | **2×** | Shared lead actors matter but are noisier |
| `director` | **3×** | Director style is a very strong thematic signal |

These weights were selected for the Phase 2 benchmark and will be
systematically tuned in a future evaluation phase.
        """
    )

# ── 5. Models ────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/model_training: Models Evaluated")

    tab_cv, tab_tfidf, tab_wtfidf = st.tabs(
        ["CountVectorizer (baseline)", "TF-IDF (production)", "TF-IDF boosted"]
    )

    with tab_cv:
        st.markdown(
            """
**CountVectorizer** converts the tags string into a sparse vector of raw token counts.
Each dimension corresponds to one vocabulary word; its value is how many times that
word appears.

- Treats all words **equally** regardless of how common they are across the corpus.
- Genre tokens like `action` or `drama` (repeated 3× and present in hundreds of films)
  dominate the similarity calculation.
- Simple, fast, and interpretable.
- Serves as the **baseline** for comparison.
            """
        )

    with tab_tfidf:
        st.markdown(
            """
**TF-IDF (Term Frequency – Inverse Document Frequency)** builds on CountVectorizer by
multiplying each term count by a penalty for how commonly it appears across all movies.

```
TF-IDF(word, movie) = count(word in movie) × log(N / docs containing word)
```

- Common words like `action` appear in many films → **down-weighted**.
- Rare, distinctive words like `inception` or `ChristopherNolan` → **up-weighted**.
- Produces tighter, more thematic matches than the baseline.
- **Selected as the production model** after benchmarking.
            """
        )

    with tab_wtfidf:
        st.markdown(
            """
**TF-IDF with boosted keywords/director** uses heavier weights
(`keywords×4`, `director×4`) to further amplify rare, specific signals.

- Stronger for director-centric queries (e.g. "films like Nolan's work").
- Risk of over-narrowing for films with thin TMDB metadata.
- Won R@5 by +0.5% over the default TF-IDF but lost P@10 by 0.8%.
- Not selected; systematic tuning deferred to the next phase.
            """
        )

# ── 6. Evaluation ────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/analytics: Benchmark Results")
    st.caption(
        "Evaluated on a manually curated ground-truth set of 25 seed movies. "
        "All numbers are real — computed by `scripts/evaluate_models.py`."
    )

    st.info(
        "**Methodology**: No real user-interaction data exists. Ground truth was created by a "
        "human curator who listed films that a domain-informed viewer would consider genuinely "
        "similar for each seed. Metrics measure relative model quality, not absolute accuracy.",
        icon=":material/info:",
    )

    # Real results from Phase 2 benchmark run
    benchmark_data = {
        "Model": [
            "CountVectorizer (baseline)",
            "**TF-IDF (production)** ✓",
            "TF-IDF boosted kw/dir",
        ],
        "Seeds evaluated": [25, 25, 25],
        "P@5": ["11.2%", "**12.0%**", "12.0%"],
        "P@10": ["8.0%", "**8.4%**", "7.6%"],
        "R@5": ["9.0%", "**9.5%**", "10.0%"],
        "R@10": ["12.8%", "**13.8%**", "12.4%"],
        "Avg latency": ["0.0017s", "**0.0015s**", "0.0014s"],
    }

    import pandas as pd
    st.dataframe(
        pd.DataFrame(benchmark_data),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
**Metric definitions**

| Metric | Meaning |
|---|---|
| Precision@K | Fraction of top-K recommendations that are in the ground-truth set |
| Recall@K | Fraction of the ground-truth set recovered in the top-K |
        """
    )

# ── 7. Selected Model ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/verified: Selected Model — TF-IDF")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
**Why TF-IDF was chosen over CountVectorizer:**

- Highest P@5 and P@10 — the primary quality metrics for a user-facing product.
- Equivalent or better recall across both K values.
- IDF normalization is a principled improvement: it suppresses the noise from
  common genre tokens that dominate CountVectorizer results.
- Latency is statistically identical (< 2 ms per query on a standard laptop).
- No memory or deployment overhead vs CountVectorizer.
            """
        )
    with col_right:
        st.markdown(
            """
**Why not the boosted TF-IDF variant:**

- +0.5% gain in R@5 does not justify the -0.8% loss in P@10.
- Heavier keyword/director weights risk over-fitting to well-documented
  blockbusters while producing poor results for films with sparse metadata.
- The default weight configuration is simpler and better understood.
- Boosted weights will be re-evaluated after systematic tuning.
            """
        )

# ── 8. Limitations ────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/warning: Limitations")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**System limitations**

- **Content-only**: The model sees metadata, not viewing behaviour.
  It cannot learn that you dislike horror or that you watch comedies
  exclusively on weekends.
- **Metadata quality**: Films with sparse or inaccurate TMDB metadata
  receive worse recommendations. Older and non-English films are
  disproportionately affected.
- **No collaborative filtering**: User-to-user similarity ("people like
  you also watched…") is not implemented.
- **No personalization**: The model produces the same results for
  any user who selects the same seed movie.
- **Static dataset**: The TMDB 5000 corpus is a snapshot. Films released
  after the dataset was collected cannot be recommended.
            """
        )
    with right:
        st.markdown(
            """
**Evaluation limitations**

- **Manually curated benchmark**: The 25-seed ground-truth set was
  created by a human curator, not from real user interaction logs.
  It reflects one person's opinion of "similar films".
- **Low absolute metrics are expected**: A P@5 of ~12% is normal for
  content-based systems without user data. The benchmark is useful
  for **comparing models**, not measuring absolute quality.
- **Small evaluation set**: 25 seeds covers a limited range of genres
  and film styles. Coverage will be expanded in future phases.
- **No A/B testing**: We have no click-through or watch-through data
  to validate that higher P@K translates to better user satisfaction.
            """
        )

st.caption(
    "CineMatch is an open-source learning project. "
    "Source code and evaluation scripts are available in the repository."
)

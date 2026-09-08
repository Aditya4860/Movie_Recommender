"""CineMatch FastAPI service.

Exposes the pre-trained TF-IDF content-based recommendation model via a
clean REST API.  The model loads once at startup from models/ and is
reused for every request — no retraining per request.

Endpoints
---------
GET /health          — liveness + model info
GET /movies          — paginated list of all titles in the corpus
GET /recommend       — top-K recommendations for a given movie title

Run locally
-----------
    uvicorn api.main:app --reload

Docs (auto-generated)
---------------------
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path when the module is launched directly.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.model_store import load_manifest, load_model
from src.recommender import MovieRecommender

# ---------------------------------------------------------------------------
# Application state — holds the singleton loaded at startup
# ---------------------------------------------------------------------------

class _AppState:
    recommender: MovieRecommender | None = None
    manifest: dict = {}

_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the persisted model into memory before accepting requests."""
    models_dir = Path("models")
    try:
        _state.recommender = load_model(models_dir)
        _state.manifest = load_manifest(models_dir)
        corpus_size = len(_state.recommender.movies)
        print(
            f"[CineMatch API] Model loaded — method={_state.manifest.get('method', 'unknown')}, "
            f"corpus={corpus_size:,} movies"
        )
    except FileNotFoundError as exc:
        print(f"[CineMatch API] FATAL: {exc}")
        raise RuntimeError(str(exc)) from exc
    yield
    # Cleanup (nothing needed for in-memory model)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CineMatch Recommendation API",
    description=(
        "Content-based movie recommendation service powered by TF-IDF cosine similarity "
        "on the TMDB 5000 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the model is ready")
    model_method: str = Field(..., description="Vectorizer method in use")
    corpus_size: int = Field(..., description="Number of movies in the corpus")
    built_at: str = Field(..., description="ISO-8601 timestamp of the last model build")


class MovieSummary(BaseModel):
    id: int = Field(..., description="TMDB movie ID")
    title: str = Field(..., description="Movie title")
    genres: list[str] = Field(default_factory=list, description="Genre list")


class RecommendationItem(BaseModel):
    rank: int = Field(..., description="1-based rank in the result list")
    id: int = Field(..., description="TMDB movie ID of the recommended film")
    title: str = Field(..., description="Recommended movie title")
    similarity: float = Field(..., description="Cosine similarity score (0–100)")
    genres: list[str] = Field(default_factory=list, description="Genre list")


class RecommendResponse(BaseModel):
    seed: MovieSummary = Field(..., description="The movie used as input")
    recommendations: list[RecommendationItem] = Field(..., description="Ordered recommendations")
    model_method: str = Field(..., description="Vectorizer method used")
    k: int = Field(..., description="Number of recommendations returned")


class MoviesResponse(BaseModel):
    total: int = Field(..., description="Total movies in the corpus")
    page: int
    page_size: int
    movies: list[MovieSummary]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model() -> MovieRecommender:
    """Return the loaded recommender or raise 503 if not ready."""
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _state.recommender


def _lookup_by_title(title: str, recommender: MovieRecommender) -> dict:
    """Case-insensitive title lookup.  Raises 404 if not found."""
    query = title.casefold().strip()
    matches = recommender.movies[recommender.movies["title"].str.casefold() == query]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{title}' was not found in the corpus. "
                   "Use GET /movies to browse available titles.",
        )
    return matches.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return API liveness status and model metadata."""
    rec = _require_model()
    return HealthResponse(
        status="ok",
        model_method=_state.manifest.get("method", rec.method),
        corpus_size=len(rec.movies),
        built_at=_state.manifest.get("built_at", "unknown"),
    )


@app.get("/movies", response_model=MoviesResponse, tags=["Catalog"])
def list_movies(
    page: Annotated[int, Query(ge=1, description="Page number (1-based)")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Results per page")] = 50,
    search: Annotated[
        str | None,
        Query(description="Optional title substring filter (case-insensitive)"),
    ] = None,
) -> MoviesResponse:
    """Return a paginated list of all movies in the corpus.

    Optionally filter by title substring using the ``search`` parameter.
    """
    rec = _require_model()
    df = rec.movies[["id", "title", "genres"]].copy()

    if search:
        df = df[df["title"].str.contains(search, case=False, na=False)]

    total = int(len(df))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return MoviesResponse(
        total=total,
        page=page,
        page_size=page_size,
        movies=[
            MovieSummary(
                id=int(row["id"]),
                title=str(row["title"]),
                genres=row["genres"] if isinstance(row["genres"], list) else [],
            )
            for _, row in page_df.iterrows()
        ],
    )


@app.get("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(
    movie: Annotated[str, Query(min_length=1, description="Seed movie title (exact or close match)")],
    k: Annotated[
        int,
        Query(ge=1, le=20, description="Number of recommendations to return (1–20)"),
    ] = 5,
) -> RecommendResponse:
    """Return top-K content-based recommendations for the given movie title.

    Lookup is **case-insensitive**.  If the title is not found a 404 is
    returned with a helpful message.

    Example
    -------
        GET /recommend?movie=Inception&k=5
    """
    rec = _require_model()
    seed_row = _lookup_by_title(movie, rec)

    seed_id = int(seed_row["id"])
    results_df = rec.recommend(seed_id, count=k)

    # Join titles and genres back from the movies frame
    merged = results_df.merge(rec.movies[["id", "title", "genres"]], on="id", how="left")

    recommendations = [
        RecommendationItem(
            rank=rank + 1,
            id=int(row["id"]),
            title=str(row["title"]),
            similarity=float(row["similarity"]),
            genres=row["genres"] if isinstance(row["genres"], list) else [],
        )
        for rank, (_, row) in enumerate(merged.iterrows())
    ]

    seed_genres = seed_row.get("genres", [])
    if not isinstance(seed_genres, list):
        seed_genres = []

    return RecommendResponse(
        seed=MovieSummary(
            id=seed_id,
            title=str(seed_row["title"]),
            genres=seed_genres,
        ),
        recommendations=recommendations,
        model_method=rec.method,
        k=k,
    )

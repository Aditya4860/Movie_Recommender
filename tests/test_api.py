"""API tests for CineMatch FastAPI service.

Uses FastAPI's TestClient (which wraps httpx) so no live server is needed.
The model is loaded from the persisted models/ directory, so
scripts/build_model.py must have been run before executing these tests.

No TMDB API calls are made — recommendations use only the pre-trained
similarity matrix.

Run with:
    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient that loads the real persisted model once."""
    from fastapi.testclient import TestClient
    from api.main import app

    models_dir = Path("models")
    if not (models_dir / "recommender.joblib").exists():
        pytest.skip("Model artifacts not found. Run `python scripts/build_model.py` first.")

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_model_method(self, client):
        data = client.get("/health").json()
        assert data["model_method"] in ("count", "tfidf")

    def test_health_has_corpus_size(self, client):
        data = client.get("/health").json()
        assert isinstance(data["corpus_size"], int)
        assert data["corpus_size"] > 0

    def test_health_has_built_at(self, client):
        data = client.get("/health").json()
        assert "built_at" in data
        assert data["built_at"] != ""

    def test_health_schema(self, client):
        data = client.get("/health").json()
        expected_keys = {"status", "model_method", "corpus_size", "built_at"}
        assert expected_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# /movies
# ---------------------------------------------------------------------------

class TestMovies:

    def test_movies_returns_200(self, client):
        assert client.get("/movies").status_code == 200

    def test_movies_default_page_size(self, client):
        data = client.get("/movies").json()
        assert len(data["movies"]) == 50  # default page_size

    def test_movies_total_is_positive(self, client):
        data = client.get("/movies").json()
        assert data["total"] > 0

    def test_movies_pagination(self, client):
        page1 = client.get("/movies?page=1&page_size=10").json()
        page2 = client.get("/movies?page=2&page_size=10").json()
        titles1 = {m["title"] for m in page1["movies"]}
        titles2 = {m["title"] for m in page2["movies"]}
        # Pages must not overlap
        assert titles1.isdisjoint(titles2)

    def test_movies_search_filter(self, client):
        data = client.get("/movies?search=batman").json()
        for movie in data["movies"]:
            assert "batman" in movie["title"].lower()

    def test_movies_search_no_results(self, client):
        data = client.get("/movies?search=zzz_definitely_not_a_film_xyz").json()
        assert data["total"] == 0
        assert data["movies"] == []

    def test_movies_item_schema(self, client):
        data = client.get("/movies?page_size=1").json()
        movie = data["movies"][0]
        assert "id" in movie
        assert "title" in movie
        assert "genres" in movie
        assert isinstance(movie["genres"], list)

    def test_movies_page_size_limit(self, client):
        # page_size > 200 should be rejected
        response = client.get("/movies?page_size=201")
        assert response.status_code == 422

    def test_movies_page_must_be_positive(self, client):
        response = client.get("/movies?page=0")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class TestRecommend:

    def test_recommend_valid_movie(self, client):
        response = client.get("/recommend?movie=Inception")
        assert response.status_code == 200

    def test_recommend_default_k(self, client):
        data = client.get("/recommend?movie=Inception").json()
        assert len(data["recommendations"]) == 5
        assert data["k"] == 5

    def test_recommend_custom_k(self, client):
        data = client.get("/recommend?movie=Inception&k=10").json()
        assert len(data["recommendations"]) == 10
        assert data["k"] == 10

    def test_recommend_seed_in_response(self, client):
        data = client.get("/recommend?movie=Inception").json()
        assert data["seed"]["title"] == "Inception"
        assert data["seed"]["id"] == 27205

    def test_recommend_response_schema(self, client):
        data = client.get("/recommend?movie=Inception").json()
        assert "seed" in data
        assert "recommendations" in data
        assert "model_method" in data
        assert "k" in data

    def test_recommend_item_schema(self, client):
        data = client.get("/recommend?movie=Inception").json()
        item = data["recommendations"][0]
        assert "rank" in item
        assert "id" in item
        assert "title" in item
        assert "similarity" in item
        assert "genres" in item

    def test_recommend_ranks_are_sequential(self, client):
        data = client.get("/recommend?movie=Inception&k=5").json()
        ranks = [r["rank"] for r in data["recommendations"]]
        assert ranks == [1, 2, 3, 4, 5]

    def test_recommend_similarity_range(self, client):
        data = client.get("/recommend?movie=Inception&k=10").json()
        for item in data["recommendations"]:
            assert 0.0 <= item["similarity"] <= 100.0

    def test_recommend_similarity_descending(self, client):
        data = client.get("/recommend?movie=Inception&k=10").json()
        scores = [r["similarity"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_seed_not_in_results(self, client):
        data = client.get("/recommend?movie=Inception").json()
        seed_id = data["seed"]["id"]
        result_ids = [r["id"] for r in data["recommendations"]]
        assert seed_id not in result_ids

    def test_recommend_case_insensitive_title(self, client):
        r1 = client.get("/recommend?movie=inception").json()
        r2 = client.get("/recommend?movie=INCEPTION").json()
        assert r1["seed"]["id"] == r2["seed"]["id"]

    def test_recommend_dark_knight(self, client):
        """The Dark Knight should return Batman films at top positions."""
        data = client.get("/recommend?movie=The Dark Knight&k=5").json()
        titles = [r["title"] for r in data["recommendations"]]
        batman_films = {"The Dark Knight Rises", "Batman Begins", "Batman Returns"}
        # At least two of the top-5 should be Batman films
        hits = batman_films.intersection(titles)
        assert len(hits) >= 2, f"Expected Batman films in top-5, got: {titles}"

    def test_recommend_invalid_movie_404(self, client):
        response = client.get("/recommend?movie=ZZZ_This_Film_Does_Not_Exist_XYZ")
        assert response.status_code == 404

    def test_recommend_invalid_movie_error_message(self, client):
        response = client.get("/recommend?movie=ZZZ_This_Film_Does_Not_Exist_XYZ")
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_recommend_k_too_large_rejected(self, client):
        response = client.get("/recommend?movie=Inception&k=21")
        assert response.status_code == 422

    def test_recommend_k_zero_rejected(self, client):
        response = client.get("/recommend?movie=Inception&k=0")
        assert response.status_code == 422

    def test_recommend_missing_movie_param(self, client):
        response = client.get("/recommend")
        assert response.status_code == 422

    def test_recommend_empty_movie_rejected(self, client):
        response = client.get("/recommend?movie=")
        assert response.status_code == 422

    def test_recommend_model_method_field(self, client):
        data = client.get("/recommend?movie=Inception").json()
        assert data["model_method"] in ("count", "tfidf")

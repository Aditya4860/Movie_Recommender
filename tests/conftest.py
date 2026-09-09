"""Shared pytest fixtures for the CineMatch test suite.

All fixtures that require ML models use a tiny synthetic dataset (10 movies)
built entirely in-memory.  No CSV files, no TMDB API calls, no internet.

The synthetic dataset is designed so that its similarity structure is
predictable:
  - Movies 1–3 are sci-fi with "space" keywords → should cluster together.
  - Movies 4–6 are crime dramas with "heist" → should cluster together.
  - Movies 7–8 are animated family films → should cluster together.
  - Movies 9–10 are romantic comedies → should cluster together.

This gives deterministic, verifiable test assertions.
"""

from __future__ import annotations

import json
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Tiny TMDB-shaped DataFrames (no disk, no internet)
# ---------------------------------------------------------------------------

def _j(items: list[dict]) -> str:
    """Serialize a list of dicts in the TMDB JSON-list-column format."""
    return json.dumps(items)


# 10 synthetic movies; column names match the real TMDB CSV
SYNTHETIC_MOVIES = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "title": [
            "Alpha Space",    # sci-fi
            "Beta Cosmos",    # sci-fi
            "Gamma Galaxy",   # sci-fi
            "Delta Heist",    # crime
            "Epsilon Vault",  # crime
            "Zeta Crime",     # crime
            "Eta Animation",  # family
            "Theta Cartoon",  # family
            "Iota Romance",   # romance
            "Kappa Love",     # romance
        ],
        "overview": [
            "Astronauts explore space and discover an alien planet.",
            "A rocket travels through the cosmos to find new worlds.",
            "Scientists investigate a distant galaxy full of stars.",
            "A crew of thieves plans a daring bank heist.",
            "Criminals break into a vault to steal diamonds.",
            "A detective uncovers a crime ring in the city.",
            "Animated animals go on a family adventure.",
            "A cartoon hero saves the world with his friends.",
            "Two strangers fall in love during a summer holiday.",
            "A romantic comedy about love found in the workplace.",
        ],
        "genres": [
            _j([{"id": 878, "name": "Science Fiction"}, {"id": 28, "name": "Action"}]),
            _j([{"id": 878, "name": "Science Fiction"}, {"id": 12, "name": "Adventure"}]),
            _j([{"id": 878, "name": "Science Fiction"}, {"id": 18, "name": "Drama"}]),
            _j([{"id": 80, "name": "Crime"}, {"id": 53, "name": "Thriller"}]),
            _j([{"id": 80, "name": "Crime"}, {"id": 18, "name": "Drama"}]),
            _j([{"id": 80, "name": "Crime"}, {"id": 53, "name": "Thriller"}]),
            _j([{"id": 16, "name": "Animation"}, {"id": 10751, "name": "Family"}]),
            _j([{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}]),
            _j([{"id": 10749, "name": "Romance"}, {"id": 35, "name": "Comedy"}]),
            _j([{"id": 10749, "name": "Romance"}, {"id": 35, "name": "Comedy"}]),
        ],
        "keywords": [
            _j([{"id": 1, "name": "space"}, {"id": 2, "name": "alien"}]),
            _j([{"id": 1, "name": "space"}, {"id": 3, "name": "rocket"}]),
            _j([{"id": 1, "name": "space"}, {"id": 4, "name": "galaxy"}]),
            _j([{"id": 5, "name": "heist"}, {"id": 6, "name": "bank"}]),
            _j([{"id": 5, "name": "heist"}, {"id": 7, "name": "vault"}]),
            _j([{"id": 5, "name": "heist"}, {"id": 8, "name": "detective"}]),
            _j([{"id": 9, "name": "animation"}, {"id": 10, "name": "adventure"}]),
            _j([{"id": 9, "name": "animation"}, {"id": 11, "name": "hero"}]),
            _j([{"id": 12, "name": "love"}, {"id": 13, "name": "holiday"}]),
            _j([{"id": 12, "name": "love"}, {"id": 14, "name": "workplace"}]),
        ],
        "release_date": ["2020-01-01"] * 10,
        "vote_average": [7.5, 7.2, 7.8, 6.9, 7.1, 6.8, 8.0, 7.6, 6.5, 6.7],
        "vote_count": [1000, 900, 1100, 800, 750, 700, 1200, 950, 600, 650],
        "popularity": [50.0] * 10,
        "runtime": [120] * 10,
    }
)

SYNTHETIC_CREDITS = pd.DataFrame(
    {
        "title": [
            "Alpha Space",
            "Beta Cosmos",
            "Gamma Galaxy",
            "Delta Heist",
            "Epsilon Vault",
            "Zeta Crime",
            "Eta Animation",
            "Theta Cartoon",
            "Iota Romance",
            "Kappa Love",
        ],
        "cast": [
            _j([{"id": 1, "name": "Alice Star"}, {"id": 2, "name": "Bob Planet"}]),
            _j([{"id": 3, "name": "Carol Moon"}, {"id": 4, "name": "Dave Sky"}]),
            _j([{"id": 5, "name": "Eve Galaxy"}, {"id": 6, "name": "Frank Orbit"}]),
            _j([{"id": 7, "name": "Grace Thief"}, {"id": 8, "name": "Hank Safe"}]),
            _j([{"id": 9, "name": "Iris Jewel"}, {"id": 10, "name": "Jake Vault"}]),
            _j([{"id": 11, "name": "Karl Crime"}, {"id": 12, "name": "Laura Law"}]),
            _j([{"id": 13, "name": "Mike Toon"}, {"id": 14, "name": "Nina Draw"}]),
            _j([{"id": 15, "name": "Oscar Hero"}, {"id": 16, "name": "Pam Save"}]),
            _j([{"id": 17, "name": "Quinn Heart"}, {"id": 18, "name": "Rick Joy"}]),
            _j([{"id": 19, "name": "Sara Love"}, {"id": 20, "name": "Tom Hug"}]),
        ],
        "crew": [
            _j([{"job": "Director", "name": "Dir SciFi", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir SciFi", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir SciFi", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Crime", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Crime", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Crime", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Anim", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Anim", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Romance", "department": "Directing"}]),
            _j([{"job": "Director", "name": "Dir Romance", "department": "Directing"}]),
        ],
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_movies() -> pd.DataFrame:
    return SYNTHETIC_MOVIES.copy()


@pytest.fixture(scope="session")
def synthetic_credits() -> pd.DataFrame:
    return SYNTHETIC_CREDITS.copy()


@pytest.fixture(scope="session")
def count_recommender(synthetic_movies, synthetic_credits):
    """A CountVectorizer recommender trained on the synthetic dataset."""
    from src.recommender import MovieRecommender
    return MovieRecommender.from_frames(
        movies=synthetic_movies,
        credits=synthetic_credits,
        method="count",
    )


@pytest.fixture(scope="session")
def tfidf_recommender(synthetic_movies, synthetic_credits):
    """A TF-IDF recommender trained on the synthetic dataset."""
    from src.recommender import MovieRecommender
    return MovieRecommender.from_frames(
        movies=synthetic_movies,
        credits=synthetic_credits,
        method="tfidf",
    )

"""Unit tests for preprocessing, MovieRecommender, and recommendation logic.

All tests use the synthetic 10-movie dataset from conftest.py.
No CSV files are read.  No TMDB API calls are made.
All assertions are deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# 1. Preprocessing helpers
# ===========================================================================

class TestParseJsonList:
    """src.data.parse_json_list — the lowest-level parsing primitive."""

    def setup_method(self):
        from src.data import parse_json_list
        self.parse = parse_json_list

    def test_valid_list_of_dicts(self):
        raw = json.dumps([{"id": 1, "name": "Action"}, {"id": 2, "name": "Drama"}])
        result = self.parse(raw)
        assert result == [{"id": 1, "name": "Action"}, {"id": 2, "name": "Drama"}]

    def test_empty_string_returns_empty_list(self):
        assert self.parse("") == []

    def test_none_returns_empty_list(self):
        assert self.parse(None) == []

    def test_whitespace_returns_empty_list(self):
        assert self.parse("   ") == []

    def test_invalid_json_returns_empty_list(self):
        assert self.parse("not json {{{") == []

    def test_dict_not_list_returns_empty(self):
        assert self.parse(json.dumps({"key": "value"})) == []

    def test_empty_list_string(self):
        assert self.parse("[]") == []

    def test_single_item(self):
        raw = json.dumps([{"id": 3, "name": "Horror"}])
        assert self.parse(raw) == [{"id": 3, "name": "Horror"}]


class TestExtractNames:
    """src.preprocessing._extract_names — token extraction and compaction."""

    def setup_method(self):
        from src.preprocessing import _extract_names
        self.fn = _extract_names

    def test_extracts_name_fields(self):
        raw = json.dumps([{"id": 1, "name": "Science Fiction"}, {"id": 2, "name": "Action"}])
        result = self.fn(raw)
        assert "ScienceFiction" in result
        assert "Action" in result

    def test_spaces_removed_from_names(self):
        raw = json.dumps([{"id": 1, "name": "Christopher Nolan"}])
        result = self.fn(raw)
        assert result == ["ChristopherNolan"]

    def test_limit_applied(self):
        raw = json.dumps([{"id": i, "name": f"Actor{i}"} for i in range(10)])
        result = self.fn(raw, limit=3)
        assert len(result) == 3

    def test_no_limit_returns_all(self):
        raw = json.dumps([{"id": i, "name": f"Actor{i}"} for i in range(6)])
        result = self.fn(raw)
        assert len(result) == 6

    def test_empty_input_returns_empty_list(self):
        assert self.fn("[]") == []
        assert self.fn("") == []
        assert self.fn(None) == []

    def test_missing_name_key_skipped(self):
        raw = json.dumps([{"id": 1, "character": "Hero"}, {"id": 2, "name": "Real Name"}])
        result = self.fn(raw)
        assert result == ["RealName"]


class TestExtractDirector:
    """src.preprocessing._extract_director."""

    def setup_method(self):
        from src.preprocessing import _extract_director
        self.fn = _extract_director

    def test_finds_director(self):
        crew = json.dumps([
            {"job": "Producer", "name": "John Producer"},
            {"job": "Director", "name": "Jane Director"},
        ])
        result = self.fn(crew)
        assert result == ["JaneDirector"]

    def test_no_director_returns_empty(self):
        crew = json.dumps([{"job": "Producer", "name": "John Producer"}])
        assert self.fn(crew) == []

    def test_empty_crew_returns_empty(self):
        assert self.fn("[]") == []
        assert self.fn("") == []

    def test_director_name_spaces_removed(self):
        crew = json.dumps([{"job": "Director", "name": "Christopher Nolan"}])
        assert self.fn(crew) == ["ChristopherNolan"]

    def test_returns_first_director_only(self):
        crew = json.dumps([
            {"job": "Director", "name": "First Director"},
            {"job": "Director", "name": "Second Director"},
        ])
        result = self.fn(crew)
        # Only one director expected (first match)
        assert len(result) == 1
        assert result[0] == "FirstDirector"


class TestBuildFeatureFrame:
    """src.preprocessing.build_feature_frame using from_frames path."""

    def test_returns_required_columns(self, synthetic_movies, synthetic_credits):
        from src.recommender import MovieRecommender
        rec = MovieRecommender.from_frames(
            movies=synthetic_movies,
            credits=synthetic_credits,
            method="count",
        )
        assert "id" in rec.movies.columns
        assert "title" in rec.movies.columns
        assert "genres" in rec.movies.columns
        assert "tags" in rec.movies.columns

    def test_corpus_size_matches_input(self, synthetic_movies, synthetic_credits):
        from src.recommender import MovieRecommender
        rec = MovieRecommender.from_frames(
            movies=synthetic_movies,
            credits=synthetic_credits,
            method="count",
        )
        assert len(rec.movies) == len(synthetic_movies)

    def test_tags_are_non_empty_strings(self, synthetic_movies, synthetic_credits):
        from src.recommender import MovieRecommender
        rec = MovieRecommender.from_frames(
            movies=synthetic_movies,
            credits=synthetic_credits,
            method="count",
        )
        assert all(isinstance(t, str) and len(t) > 0 for t in rec.movies["tags"])

    def test_genres_are_lists(self, synthetic_movies, synthetic_credits):
        from src.recommender import MovieRecommender
        rec = MovieRecommender.from_frames(
            movies=synthetic_movies,
            credits=synthetic_credits,
            method="count",
        )
        assert all(isinstance(g, list) for g in rec.movies["genres"])

    def test_missing_columns_raises_value_error(self):
        import pandas as pd
        from src.recommender import MovieRecommender
        bad_movies = pd.DataFrame({"id": [1], "title": ["X"]})  # missing columns
        bad_credits = pd.DataFrame({"title": ["X"], "cast": ["[]"], "crew": ["[]"]})
        with pytest.raises(ValueError, match="incomplete"):
            MovieRecommender.from_frames(movies=bad_movies, credits=bad_credits)


# ===========================================================================
# 2. MovieRecommender — structure
# ===========================================================================

class TestRecommenderStructure:

    def test_similarity_matrix_shape(self, count_recommender):
        n = len(count_recommender.movies)
        assert count_recommender.similarity.shape == (n, n)

    def test_similarity_matrix_dtype(self, count_recommender):
        import numpy as np
        assert count_recommender.similarity.dtype == np.float32

    def test_similarity_diagonal_is_one(self, count_recommender):
        import numpy as np
        diag = count_recommender.similarity.diagonal()
        assert all(abs(v - 1.0) < 1e-5 for v in diag), "Self-similarity should be ~1.0"

    def test_similarity_is_symmetric(self, count_recommender):
        import numpy as np
        sim = count_recommender.similarity
        assert np.allclose(sim, sim.T, atol=1e-5)

    def test_method_attribute_count(self, count_recommender):
        assert count_recommender.method == "count"

    def test_method_attribute_tfidf(self, tfidf_recommender):
        assert tfidf_recommender.method == "tfidf"

    def test_titles_property_sorted(self, count_recommender):
        titles = count_recommender.titles
        assert titles == sorted(titles)

    def test_titles_property_length(self, count_recommender):
        assert len(count_recommender.titles) == len(count_recommender.movies)


# ===========================================================================
# 3. Movie lookup by ID
# ===========================================================================

class TestMovieLookup:

    def test_lookup_valid_id_returns_dataframe(self, count_recommender):
        result = count_recommender.recommend(1, 3)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_lookup_invalid_id_raises(self, count_recommender):
        with pytest.raises(ValueError, match="not found"):
            count_recommender.recommend(9999, 3)

    def test_lookup_zero_id_raises(self, count_recommender):
        with pytest.raises(ValueError, match="not found"):
            count_recommender.recommend(0, 3)

    def test_lookup_negative_id_raises(self, count_recommender):
        with pytest.raises(ValueError, match="not found"):
            count_recommender.recommend(-1, 5)

    def test_seed_not_in_results(self, count_recommender):
        results = count_recommender.recommend(1, 5)
        assert 1 not in results["id"].tolist()


# ===========================================================================
# 4. CountVectorizer recommendations
# ===========================================================================

class TestCountVectorizerRecommendations:
    """Verify clustering behaviour of CountVectorizer on the synthetic dataset."""

    def test_scifi_clusters_together(self, count_recommender):
        """Alpha Space (id=1) should recommend Beta Cosmos (2) and Gamma Galaxy (3)."""
        results = count_recommender.recommend(1, 3)
        result_ids = set(results["id"].tolist())
        scifi_ids = {2, 3}
        assert scifi_ids.issubset(result_ids), (
            f"Sci-fi movies should cluster; got IDs {result_ids}"
        )

    def test_crime_clusters_together(self, count_recommender):
        """Delta Heist (id=4) should recommend Epsilon Vault (5) and Zeta Crime (6)."""
        results = count_recommender.recommend(4, 3)
        result_ids = set(results["id"].tolist())
        crime_ids = {5, 6}
        assert crime_ids.issubset(result_ids), (
            f"Crime movies should cluster; got IDs {result_ids}"
        )

    def test_result_count_respected(self, count_recommender):
        for k in [1, 2, 3, 5]:
            results = count_recommender.recommend(1, k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    def test_similarity_scores_descending(self, count_recommender):
        results = count_recommender.recommend(1, 5)
        scores = results["similarity"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_similarity_score_range(self, count_recommender):
        results = count_recommender.recommend(1, 9)
        assert all(0.0 <= s <= 100.0 for s in results["similarity"])

    def test_result_has_id_column(self, count_recommender):
        results = count_recommender.recommend(1, 3)
        assert "id" in results.columns

    def test_result_has_similarity_column(self, count_recommender):
        results = count_recommender.recommend(1, 3)
        assert "similarity" in results.columns

    def test_result_ids_are_integers(self, count_recommender):
        results = count_recommender.recommend(1, 3)
        for movie_id in results["id"]:
            assert isinstance(movie_id, (int,)), f"Expected int, got {type(movie_id)}"


# ===========================================================================
# 5. TF-IDF recommendations
# ===========================================================================

class TestTfIdfRecommendations:
    """Same logical guarantees for the TF-IDF model."""

    def test_scifi_clusters_together(self, tfidf_recommender):
        results = tfidf_recommender.recommend(1, 3)
        result_ids = set(results["id"].tolist())
        scifi_ids = {2, 3}
        assert scifi_ids.issubset(result_ids), (
            f"Sci-fi movies should cluster under TF-IDF; got IDs {result_ids}"
        )

    def test_crime_clusters_together(self, tfidf_recommender):
        results = tfidf_recommender.recommend(4, 3)
        result_ids = set(results["id"].tolist())
        crime_ids = {5, 6}
        assert crime_ids.issubset(result_ids), (
            f"Crime movies should cluster under TF-IDF; got IDs {result_ids}"
        )

    def test_result_count_respected(self, tfidf_recommender):
        for k in [1, 2, 3, 5]:
            results = tfidf_recommender.recommend(1, k)
            assert len(results) == k

    def test_similarity_descending(self, tfidf_recommender):
        results = tfidf_recommender.recommend(1, 5)
        scores = results["similarity"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_similarity_score_range(self, tfidf_recommender):
        results = tfidf_recommender.recommend(1, 9)
        assert all(0.0 <= s <= 100.0 for s in results["similarity"])

    def test_method_is_tfidf(self, tfidf_recommender):
        assert tfidf_recommender.method == "tfidf"

    def test_seed_not_in_results(self, tfidf_recommender):
        results = tfidf_recommender.recommend(7, 5)
        assert 7 not in results["id"].tolist()


# ===========================================================================
# 6. Recommendation count parameter
# ===========================================================================

class TestRecommendationCount:

    @pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 9])
    def test_count_returned_equals_k(self, count_recommender, k):
        results = count_recommender.recommend(1, k)
        assert len(results) == k

    def test_count_cannot_exceed_corpus_minus_one(self, count_recommender):
        """Requesting more than n-1 should return at most n-1 results."""
        corpus_size = len(count_recommender.movies)
        results = count_recommender.recommend(1, corpus_size + 100)
        assert len(results) <= corpus_size - 1

    def test_count_one_returns_single_row(self, count_recommender):
        results = count_recommender.recommend(1, 1)
        assert len(results) == 1

    def test_count_parameter_consistent_across_methods(
        self, count_recommender, tfidf_recommender
    ):
        for k in [1, 3, 5]:
            assert len(count_recommender.recommend(1, k)) == k
            assert len(tfidf_recommender.recommend(1, k)) == k


# ===========================================================================
# 7. Invalid movie handling
# ===========================================================================

class TestInvalidMovieHandling:

    def test_unknown_id_raises_value_error(self, count_recommender):
        with pytest.raises(ValueError):
            count_recommender.recommend(99999, 5)

    def test_unknown_id_error_message(self, count_recommender):
        with pytest.raises(ValueError, match="99999"):
            count_recommender.recommend(99999, 5)

    def test_string_id_raises(self, count_recommender):
        """Passing a non-integer ID should raise — type mismatch with the int column."""
        with pytest.raises(Exception):
            count_recommender.recommend("not_an_id", 5)

    def test_tfidf_unknown_id_raises(self, tfidf_recommender):
        with pytest.raises(ValueError):
            tfidf_recommender.recommend(99999, 5)

    def test_error_mentions_id_in_message(self, count_recommender):
        bad_id = 12345
        with pytest.raises(ValueError) as exc_info:
            count_recommender.recommend(bad_id, 5)
        assert str(bad_id) in str(exc_info.value)

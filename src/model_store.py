"""Model artifact persistence for CineMatch.

Handles saving and loading a fitted MovieRecommender to/from disk so that
the FastAPI service can start instantly without retraining.

Serialization format
--------------------
We use joblib (ships with scikit-learn) to pickle the model.
A single file ``models/recommender.joblib`` stores the whole object.

joblib is preferred over pickle for numpy arrays (the similarity matrix)
because it uses memory-mapped I/O for large arrays, reducing load time
and peak RSS.

The movies DataFrame is stored as a Parquet sidecar
``models/movies.parquet`` because:
  - Parquet is schema-aware and type-safe (avoids pickle subtleties with pandas).
  - It is fast and compact for columnar tabular data.
  - It separates the ML matrix (joblib) from metadata (Parquet) cleanly.

Artifact layout
---------------
models/
    recommender.joblib   — similarity matrix + method + model metadata
    movies.parquet       — preprocessed movie catalogue (id, title, genres, tags)
    manifest.json        — build timestamp, method, weights, corpus size
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Avoid a hard dependency on the full recommender at import time
# (tests can import model_store without needing the full ML stack).
MODELS_DIR = Path("models")

_MATRIX_FILE = "recommender.joblib"
_MOVIES_FILE = "movies.parquet"
_MANIFEST_FILE = "manifest.json"


def _ensure_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def save_model(
    recommender,  # MovieRecommender
    models_dir: str | Path = MODELS_DIR,
    weights: dict[str, int] | None = None,
) -> Path:
    """Persist a fitted MovieRecommender to ``models_dir``.

    Parameters
    ----------
    recommender:
        A fitted ``MovieRecommender`` instance.
    models_dir:
        Directory to write artifacts into.  Created if absent.
    weights:
        The feature weights used during training (recorded in the manifest).

    Returns
    -------
    Path to the manifest JSON file.
    """
    out = Path(models_dir)
    _ensure_models_dir(out)

    # 1. Similarity matrix + method → joblib
    matrix_path = out / _MATRIX_FILE
    payload = {
        "similarity": recommender.similarity,
        "method": recommender.method,
    }
    joblib.dump(payload, matrix_path, compress=3)

    # 2. Movie metadata DataFrame → Parquet
    movies_path = out / _MOVIES_FILE
    # genres column holds list[str]; convert to JSON string for Parquet compat
    movies_out = recommender.movies.copy()
    movies_out["genres"] = movies_out["genres"].map(json.dumps)
    movies_out.to_parquet(movies_path, index=False)

    # 3. Manifest
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "method": recommender.method,
        "weights": weights or {},
        "corpus_size": int(len(recommender.movies)),
        "matrix_shape": list(recommender.similarity.shape),
        "matrix_dtype": str(recommender.similarity.dtype),
        "artifacts": {
            "matrix": _MATRIX_FILE,
            "movies": _MOVIES_FILE,
        },
    }
    manifest_path = out / _MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[model_store] Saved to {out.resolve()}")
    print(f"  Matrix : {matrix_path.name}  ({matrix_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Movies : {movies_path.name}  ({movies_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Manifest: {manifest_path.name}")

    return manifest_path


def load_model(models_dir: str | Path = MODELS_DIR):
    """Load a previously persisted MovieRecommender from ``models_dir``.

    Returns
    -------
    A ``MovieRecommender`` instance reconstructed from disk.

    Raises
    ------
    FileNotFoundError
        If the expected artifact files are missing.
    """
    from src.recommender import MovieRecommender  # local import avoids circularity

    out = Path(models_dir)
    matrix_path = out / _MATRIX_FILE
    movies_path = out / _MOVIES_FILE

    for path in (matrix_path, movies_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. "
                "Run `python scripts/build_model.py` to generate it."
            )

    # Load similarity matrix + method
    payload = joblib.load(matrix_path)
    similarity: np.ndarray = payload["similarity"]
    method: str = payload["method"]

    # Load movie metadata
    movies = pd.read_parquet(movies_path)
    # Restore genres from JSON string back to list[str]
    movies["genres"] = movies["genres"].map(json.loads)

    return MovieRecommender(movies=movies, similarity=similarity, method=method)


def load_manifest(models_dir: str | Path = MODELS_DIR) -> dict:
    """Return the saved manifest dict, or an empty dict if not found."""
    path = Path(models_dir) / _MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())

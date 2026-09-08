"""One-time script to train and persist the selected CineMatch model.

The selected model (TF-IDF with default weights) is built from the TMDB
CSV files and saved to models/ so the FastAPI service can start instantly
without retraining.

Usage
-----
    python scripts/build_model.py [--data-dir data] [--models-dir models]

This script only needs to be re-run when:
  - The dataset changes.
  - The preprocessing logic changes.
  - The feature weights change.
  - The vectorizer method changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_store import save_model
from src.preprocessing import DEFAULT_WEIGHTS
from src.recommender import MovieRecommender


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and persist the CineMatch recommendation model."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing tmdb_5000_movies.csv and tmdb_5000_credits.csv",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to write model artifacts into (default: models/)",
    )
    parser.add_argument(
        "--method",
        choices=["count", "tfidf"],
        default="tfidf",
        help="Vectorizer method to use (default: tfidf — recommended by benchmark)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CineMatch — Model Build")
    print(f"  Method      : {args.method}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Models dir  : {args.models_dir}")
    print(f"  Weights     : {DEFAULT_WEIGHTS}")
    print("=" * 60)

    print("\n[1/2] Training model…")
    t0 = time.perf_counter()
    recommender = MovieRecommender.from_csv(
        data_dir=args.data_dir,
        method=args.method,
        weights=None,  # uses DEFAULT_WEIGHTS from preprocessing.py
    )
    elapsed = time.perf_counter() - t0
    print(f"      Done in {elapsed:.1f}s  |  corpus: {len(recommender.movies):,} movies")

    print("\n[2/2] Saving artifacts…")
    save_model(
        recommender=recommender,
        models_dir=args.models_dir,
        weights=DEFAULT_WEIGHTS,
    )

    print("\n[OK] Build complete. Start the API with:")
    print("    uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()

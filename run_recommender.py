"""Small command-line interface for testing recommendations."""

import argparse

from src.recommender import MovieRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Get TMDB movie recommendations.")
    parser.add_argument("title", help="Movie title to use as the starting point")
    parser.add_argument("--count", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--data-dir", default="data", help="Folder containing TMDB CSV files")
    args = parser.parse_args()

    recommender = MovieRecommender.from_csv(args.data_dir)
    print(recommender.recommend(args.title, args.count).to_string(index=False))


if __name__ == "__main__":
    main()

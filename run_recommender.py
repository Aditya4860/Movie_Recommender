"""Small command-line interface for testing recommendations."""

import argparse

from src.recommender import MovieRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Get TMDB movie recommendations.")
    parser.add_argument("title", help="Movie title to use as the starting point")
    parser.add_argument("--count", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--data-dir", default="data", help="Folder containing TMDB CSV files")
    parser.add_argument(
        "--method",
        choices=["count", "tfidf"],
        default="count",
        help="Vectorizer method: 'count' (CountVectorizer, default) or 'tfidf' (TF-IDF)",
    )
    args = parser.parse_args()

    print(f"Building recommender using method='{args.method}'…")
    recommender = MovieRecommender.from_csv(args.data_dir, method=args.method)

    query = args.title.casefold().strip()
    matches = recommender.movies[recommender.movies["title"].str.casefold() == query]

    if matches.empty:
        print(f"Movie '{args.title}' was not found in the dataset.")
        return

    movie_id = matches.iloc[0]["id"]
    results = recommender.recommend(movie_id, args.count)

    # Merge titles back for display
    results = results.merge(recommender.movies[["id", "title"]], on="id")
    print(f"\nTop {args.count} recommendations for '{args.title}' [{args.method}]:\n")
    print(results[["title", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()

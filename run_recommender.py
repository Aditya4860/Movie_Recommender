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
    query = args.title.casefold().strip()
    matches = recommender.movies[recommender.movies["title"].str.casefold() == query]
    
    if matches.empty:
        print(f"Movie '{args.title}' was not found in the dataset.")
        return
        
    movie_id = matches.iloc[0]["id"]
    results = recommender.recommend(movie_id, args.count)
    
    # Merge titles back for display
    results = results.merge(recommender.movies[["id", "title"]], on="id")
    print(results[["title", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()

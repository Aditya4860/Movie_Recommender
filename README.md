# Movie Recommendation System 🎬

A content-based movie recommendation system built with the TMDB 5000 Movie Dataset. It suggests films with similar genres, plot keywords, leading cast members, and directors.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

## About the project

This project was created as a BTech machine-learning learning project. It demonstrates a complete content-based recommendation workflow: turning raw movie metadata into features, comparing those features mathematically, and presenting useful recommendations in a simple web interface.

## Features

- **Content-based filtering** using genres, overview, keywords, cast, and director
- **Cosine similarity** to rank the closest movies
- **Interactive Streamlit app** with searchable movie selection
- **Command-line interface** for quick testing
- Accepts either standard Kaggle TMDB filenames or `movies.csv` / `credits.csv`

## Project structure

```text
movie-recommendation-system/
├── data/                         # Downloaded TMDB CSV files go here
├── src/
│   └── recommender.py             # Feature processing and recommendation engine
├── app.py                         # Streamlit interface
├── run_recommender.py             # Command-line interface
├── requirements.txt
└── README.md
```

## Dataset

Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place these two files in `data/`:

- `tmdb_5000_movies.csv` (or `movies.csv`)
- `tmdb_5000_credits.csv` (or `credits.csv`)

The dataset includes metadata such as genres, plot overview, keywords, budget, revenue, cast, and crew. Dataset files are intentionally excluded from Git because they are large and externally hosted.

## How it works

1. Movie and credit records are merged by title.
2. The overview, genres, keywords, top three cast members, and director are cleaned and combined into one text field.
3. `CountVectorizer` converts the text features to numerical vectors.
4. Cosine similarity scores every movie pair.
5. The app returns the highest-scoring similar movies.

## Getting started

### 1. Clone and install

```bash
git clone <your-repository-url>
cd movie-recommendation-system
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

### 2. Add the data

Place the two TMDB CSV files in the `data/` directory as described above.

### 3. Run the web app

```bash
streamlit run app.py
```

Streamlit will open the recommender in your browser. Select a title and choose how many recommendations to display.

### Optional: run in the terminal

```bash
python run_recommender.py "The Dark Knight" --count 5
```

Example output:

```text
                  title  similarity
          The Dark Knight Rises        57.1
                   Batman Begins        54.3
```

## Technologies used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Future improvements

- Add movie posters through the TMDB API
- Include ratings and user feedback
- Add collaborative filtering and a hybrid model
- Deploy the application to Streamlit Community Cloud

## Acknowledgments

- [TMDB](https://www.themoviedb.org/) for movie metadata
- The [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

Developed as part of a BTech learning journey in Data Science and Machine Learning.

"""External API integrations for CineMatch."""

from __future__ import annotations

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit secrets take precedence if deployed, otherwise fallback to .env
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")


@st.cache_data(show_spinner=False, ttl=86400) # Cache for a day
def fetch_poster_url(movie_id: int) -> str | None:
    """Fetch the TMDB poster URL for a given movie ID."""
    if not TMDB_API_KEY:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (requests.RequestException, ValueError):
        pass
        
    return None

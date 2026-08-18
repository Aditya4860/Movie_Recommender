"""Transparent explanation of the recommendation model."""

import streamlit as st


st.title("About the model")
st.caption("A transparent content-based recommendation system built for learning.")

with st.container(border=True):
    st.subheader("How CineMatch makes a match")
    st.markdown(
        """
1. TMDB movie and credit records are merged by title.
2. Plot overview, genres, keywords, three lead cast members, and the director are combined.
3. `CountVectorizer` converts this text into a 5,000-feature representation.
4. Cosine similarity ranks the closest films to your selected title.
        """
    )

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("What it is good at")
        st.write("Finding films with similar themes, creative teams, cast patterns, and genre signals—even when two titles are released years apart.")
with right:
    with st.container(border=True):
        st.subheader("Current limitations")
        st.write("It does not know your watch history, mood, or ratings. It may also favour films with more detailed TMDB metadata.")

st.header("Next steps")
st.write("A future version could add poster artwork through a secured TMDB API key, collaborative filtering, user ratings, and a hybrid recommendation model.")

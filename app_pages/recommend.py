"""Recommendation studio page."""

import streamlit as st

from src.ui import get_catalog, get_recommender, movie_row


st.title("Recommendation studio")
st.caption("Find films that share meaningful metadata—not just the same genre.")

catalog = get_catalog()
recommender = get_recommender()
default_title = st.session_state.get("recommendation_seed")
default_index = recommender.titles.index(default_title) if default_title in recommender.titles else None

with st.form("recommendation_form", border=True):
    selected_title = st.selectbox("What did you enjoy?", recommender.titles, index=default_index, placeholder="Search by movie title…")
    result_count = st.segmented_control("Number of results", [5, 8, 10], default=5)
    submitted = st.form_submit_button("Build recommendations", type="primary", icon=":material/auto_awesome:")

if submitted and selected_title:
    st.session_state.recommendation_seed = selected_title

chosen = st.session_state.get("recommendation_seed")
if not chosen:
    st.info("Choose a title and build recommendations to begin.", icon=":material/lightbulb:")
    st.stop()

source = catalog.loc[catalog["title"].eq(chosen)].iloc[0]
with st.container(border=True):
    st.markdown(f"### Starting from: {chosen}")
    st.caption(f"{source['genre_label']}  ·  :material/star: {source['vote_average']:.1f}/10")
    st.write(source["overview"])

result_count = int(result_count or 5)
with st.status("Comparing story, genres, cast, keywords, and director…", expanded=False) as status:
    results = recommender.recommend(chosen, result_count)
    status.update(label="Recommendations ready", state="complete", expanded=False)

st.header("Your matches")
for rank, result in results.reset_index(drop=True).iterrows():
    movie = catalog.loc[catalog["title"].eq(result["title"])].iloc[0].copy()
    movie["overview"] = f"{movie['overview']}\n\nSimilarity match: {result['similarity']}%"
    st.caption(f"MATCH {rank + 1}")
    movie_row(movie)

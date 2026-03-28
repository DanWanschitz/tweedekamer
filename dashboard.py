# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Parliamentary Speech Sentiment Dashboard")

df = pd.read_csv("data/processed/speeches_labeled.csv")
df["speech_begin"] = pd.to_datetime(df["speech_begin"], errors="coerce")

politicians = sorted(df["politician_name"].dropna().unique())
parties = sorted(df["party"].dropna().unique())

selected_politicians = st.sidebar.multiselect("Politician", politicians)
selected_parties = st.sidebar.multiselect("Party", parties)
selected_outcome = st.sidebar.selectbox("Motion outcome", ["all", "passed", "not passed"])

if selected_politicians:
    df = df[df["politician_name"].isin(selected_politicians)]
if selected_parties:
    df = df[df["party"].isin(selected_parties)]
if selected_outcome == "passed":
    df = df[df["motion_passed"] == 1]
elif selected_outcome == "not passed":
    df = df[df["motion_passed"] == 0]

sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
df["sentiment_num"] = df["sentiment_label"].map(sentiment_map)

col1, col2 = st.columns(2)

with col1:
    chart = df.groupby("politician_name", as_index=False)["sentiment_num"].mean()
    fig = px.bar(chart, x="politician_name", y="sentiment_num", title="Average Sentiment by Politician")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    chart = df.groupby(["politician_name", "sentiment_label"], as_index=False).size()
    fig = px.bar(chart, x="politician_name", y="size", color="sentiment_label", barmode="stack",
                 title="Sentiment Distribution by Politician")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Average Sentiment by Hour of Day")
chart = df.groupby("hour_of_day", as_index=False)["sentiment_num"].mean()
fig = px.line(chart, x="hour_of_day", y="sentiment_num", markers=True)
st.plotly_chart(fig, use_container_width=True)
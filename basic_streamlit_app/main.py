import streamlit as st
import pandas as pd

#title
st.title("College Basketball Streamlit App")

#short description
st.write(
    "This Streamlit app loads a dataset from a CSV file of Division I college basketball teams from games this season so far."
)

# data
df = pd.read_csv("basic_streamlit_app/data/cbb25.csv")

st.subheader("All teams, full dataset")
st.dataframe(df)


# team filter
st.subheader("Specific Team Data")
Team = st.selectbox("Select a team:", df["Team"].unique())
team_df = df[df["Team"] == Team]


st.dataframe(team_df)

# conference filter

st.subheader("Filtered by Conference")
CONF = st.multiselect("Conference", df["CONF"].unique())



if CONF:
    conf_df = df[df["CONF"].isin(CONF)]


st.dataframe(conf_df)

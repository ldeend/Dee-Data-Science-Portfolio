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


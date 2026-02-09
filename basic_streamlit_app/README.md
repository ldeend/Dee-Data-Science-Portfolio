# College Basketball Streamlit App

This Streamlit app allows users to explore the Division I men's college basketball teams and their various basic and advanced stats for this season. This is part of my semester-long Data Science Portfolio showcasing my data science skills. This app uses the dataset in basic_streamlit_app/data/cbb25.csv, which is obtained from Kaggle at this link: [College Basketball Dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/data).


## Features

- Loads a dataset of teams from the CSV file and their stats
- Displays the full dataset in a large, interactive table
- 3 sub-tables with different interactive filters
  - **Team** with a dropdown filter
  - **Conference** with a multi-select filter
  - **Number of Wins** with a range slider filter
- The tables will dynamically update based on filter selections




## How to Run the App

From the root of the repository, run:

```bash
streamlit run basic_streamlit_app/main.py
```

This app also requires both pandas and streamlit. To download, run: 

```bash
pip install streamlit pandas
```

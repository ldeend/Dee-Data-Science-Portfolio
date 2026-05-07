# College Basketball Streamlit App

An interactive Streamlit application for exploring Division I men's college basketball stats. The dataset is already loaded and described below. Use the filters to explore teams, conferences, and win totals for the current season. Run locally using the Setup Instructions or follow this link:

**[Live College Basketball App](https://cbb-basic-app.streamlit.app/)**


## Project Overview

In this project, I built a live, interactive web application with Python and Streamlit where users can
- Browse the full dataset of all Division I men's CBB teams and their stats for the season
- Filter down to a specific team with a dropdown
- Filter by one or multiple conferences with a multi-select
- Filter by number of wins with a range slider

There is no setup or coding experience required to use the app.

## Setup Instructions (How to Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/ldeend/Dee-Data-Science-Portfolio.git
cd Dee-Data-Science-Portfolio/basic_streamlit_app
```

### 2. Install dependencies

```bash
pip install streamlit pandas
```

### 3. Launch the app

From the root of the repository, run:

```bash
streamlit run basic_streamlit_app/main.py
```

This will launch the app in your browser at `http://localhost:8501/`.


## Dataset Description

The dataset `cbb25.csv` contains stats for all Division I men's college basketball teams for the 2024–25 season. It is sourced from the [College Basketball Dataset on Kaggle](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/data).

Each row is one team, and the columns cover:

| Column | What it means |
|--------|--------------|
| **Team** | Team name |
| **CONF** | Conference (e.g. ACC, Big Ten, SEC) |
| **G** | Games played |
| **W** | Wins |
| **ADJOE** | Adjusted Offensive Efficiency — points scored per 100 possessions, adjusted for opponent |
| **ADJDE** | Adjusted Defensive Efficiency — points allowed per 100 possessions, adjusted for opponent |
| **EFG_O / EFG_D** | Effective field goal percentage for offense and defense |
| **TOR / TORD** | Turnover rate on offense and defense |
| **ORB / DRB** | Offensive and defensive rebounding rate |
| **FTR / FTRD** | Free throw rate on offense and defense |
| **2P_O / 2P_D** | Two-point shooting percentage on offense and defense |
| **3P_O / 3P_D** | Three-point shooting percentage on offense and defense |
| **ADJ_T** | Adjusted tempo — estimated possessions per 40 minutes |
| **WAB** | Wins Above Bubble — how many more wins the team has than a typical bubble team with the same schedule |


## App Features


- Full Dataset View: Loads the full dataset of all Division I teams in an interactive, sortable table
- Team Filter: Dropdown to select any specific team and see only that team's row
- Conference Filter: Multi-select to choose one or multiple conferences at once — all conferences are selected by default, and you can narrow down from there
- Wins Filter: Range slider to filter teams by wins — drag either end to set a minimum and maximum win total


## Example Screenshots

**Here's what the full dataset table looks like**
<img width="726" height="468" alt="Screenshot 2026-05-07 at 4 39 16 AM" src="https://github.com/user-attachments/assets/535bcf8f-ac5e-460b-b1d2-670fbe589312" />


**Here's what the wins filter in use looks like**
<img width="798" height="588" alt="Screenshot 2026-05-07 at 4 40 15 AM" src="https://github.com/user-attachments/assets/46552268-89d6-4fc3-bf5c-2b9fa08565fe" />



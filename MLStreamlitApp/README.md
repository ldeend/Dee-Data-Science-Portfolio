# Supervised Machine Learning Explorer

An interactive Streamlit application for exploring supervised machine learning models. Upload a CSV dataset, choose a model, tune the hyperparameters, and see how your choices affect model performance.

**[Live Supervised ML App]((https://lucasdee-mlapp.streamlit.app/))**


## Project Overview

In this project, I built a live, interactive web application with Python and Streamlit where users can: 
- Upload a dataset
- Train a supervised machine learning model on it (either Logistic Regression or Linear Regression, depending on the target variable used). 
- Tune hyperparameters
- Change the train/test split

There is no coding experience required, and all machine learning topics are explained in the app.

## Setup Instructions (How to Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/ldeend/Dee-Data-Science-Portfolio.git
cd Dee-Data-Science-Portfolio/MLStreamlitApp
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

Or, a `requirements.txt` is included in this folder for Streamlit Cloud deployment.

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run mlapp.py
```

This will launch the app in your browser at the IP `http://localhost:8501/`.


## App Features



## References



# Unsupervised Machine Learning Explorer

An interactive Streamlit application for exploring unsupervised machine learning models. Upload a CSV dataset, choose a model, tune the hyperparameters, and see how your choices affect model performance.

**[Live Unsupervised ML App](https://lucasdee-unsupervisedmlapp.streamlit.app/)**


## Project Overview

In this project, I built a live, interactive web application with Python and Streamlit where users can:
- Upload a dataset
- Train an unsupervised machine learning model on it (K-Means Clustering, Hierarchical Clustering, or PCA, depending on what the user wants to explore)
- Tune hyperparameters
- Observe how changes affect clustering structure and variance explained

There is no coding experience required, and all machine learning topics are explained in the app.

## Setup Instructions (How to Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/ldeend/Dee-Data-Science-Portfolio.git
cd Dee-Data-Science-Portfolio/MLUnsupervisedApp
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy
```

Or, a `requirements.txt` is included in this folder for Streamlit Cloud deployment.

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run usmlapp.py
```

This will launch the app in your browser at the IP `http://localhost:8501/`.



## App Features

### Dataset Upload and View
- Upload a CSV file through the Upload/Drop on the sidebar
- In the main panel, a preview of the first 12 rows, column names and types, and descriptive statistics
- Rows with missing values are dropped before the model is trained on it, and the app states how many were removed
- Text columns are encoded automatically using ordinal encoding, and everything is scaled properly before modeling

### Model Selection
- Select as many feature (X) variables as desired. In unsupervised learning, no target variable is used.
- Choose between K-Means Clustering, Hierarchical Clustering, and PCA
- You can choose values for each model's hyperparameters and try different random seeds
- Every time an aspect of the model is changed (tweaking a hyperparameter, adding a feature, etc.), it re-renders the model automatically, allowing for quick comparison and tuning

## References

For more information or clarification about some of the machine learning topics used, here are some references I used:
- [What is K-Means Clustering?](https://www.youtube.com/watch?v=4b5d3muPQmA&t=18s)
- [What is Hierarchical Clustering?](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [What is PCA and how is it used for dimensionality reduction?](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [What is an Elbow Plot?](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)
- [What is a Dendrogram??](https://www.geeksforgeeks.org/scipy-cluster-hierarchy-dendrogram/)

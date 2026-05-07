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

### Output
- Each model outputs variance evaluation metrics and visualizations to understand the model's performance
- For each model output, there is a dropdown panel titled **"How to interpret these metrics"** that explains what each metric is and what a good value is for each
- Along with each visualization is a short description
- Every time an aspect of the model is changed (tweaking a hyperparameter, adding a feature, etc.), it re-renders the model automatically, allowing for quick comparison and tuning 


## References

For more information or clarification about some of the machine learning topics used, here are some references I used:
- [What is K-Means Clustering?](https://www.youtube.com/watch?v=4b5d3muPQmA&t=18s)
- [What is Hierarchical Clustering?](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [What is PCA and how is it used for dimensionality reduction?](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [What is an Elbow Plot?](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)
- [What is a Dendrogram??](https://www.geeksforgeeks.org/scipy-cluster-hierarchy-dendrogram/)


## Example Screenshots

For these example screenshots, I used the men's college basketball data from 2013 from [this Kaggle link](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/data). 

**Example Dataset Preview:**
<img width="985" height="472" alt="Screenshot 2026-04-22 at 8 06 17 PM" src="https://github.com/user-attachments/assets/d50bf049-9798-4fc1-923c-7ea0cb0a8466" />

**Example Evaluation Metrics Panel:**
<img width="1032" height="392" alt="Screenshot 2026-05-07 at 6 40 39 PM" src="https://github.com/user-attachments/assets/7029e3e7-60f1-44e5-adc3-d82c939e627c" />


**Example Performance Visualization:**
<img width="885" height="759" alt="Screenshot 2026-05-07 at 6 45 32 PM" src="https://github.com/user-attachments/assets/26d5dbf0-b357-4e01-839a-31c36df8150e" />

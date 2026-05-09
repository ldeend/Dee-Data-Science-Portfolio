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


## Breakdown by Model

### K-Means Clustering

| Hyperparameter | What it does |
|----------------|-------------|
| **Number of clusters k** | The number of clusters to form. Use the Elbow Plot tab to choose the best k. |
| **Number of initializations** | How many times the algorithm runs with different seeds. The best result is kept. |
| **Max iterations** | Maximum iterations per run. Increase if the model is not converging. |

**Output metrics:** Clusters (k), WCSS, Silhouette Score
- WCSS: Sum of squared distances from each point to its cluster center. Lower = tighter clusters. Compare across different values of k.
- Silhouette Score: How similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good.
- Adjust k and watch both metrics together to find the best number of clusters.

**Visualizations:** Elbow Plot, Silhouette Plot, Scatter Plot
- Elbow Plot: Look for the 'elbow' aka the point where inertia starts decreasing more slowly. The red dashed line marks your current k. Try adjusting k to where the curve bends.
- Silhouette Plot: Wider bands = more data points in that cluster. Bands past the red average line = well-separated clusters. Thin or negative bands suggest that cluster overlaps with another.
- Scatter Plot: Each color represents a cluster. X marks show centroids. Data is projected to 2D for visualization, and axis labels show how much variance each direction captures.


### Hierarchical Clustering

| Hyperparameter | What it does |
|----------------|-------------|
| **Number of clusters k** | Where to 'cut' the dendrogram. The Dendrogram tab helps you choose this visually. |
| **Linkage method** | ward: minimizes variance within clusters. Best general purpose choice. complete: uses the maximum distance between cluster members. average: uses the mean distance between all pairs. single: uses the minimum distance, which is prone to chaining. |
| **Distance metric** | Euclidean (fixed for ward linkage). How distance between points is measured. |

**Output metrics:** Clusters (k), Linkage, Silhouette Score
- Clusters (k): Where the dendrogram is "cut." Use the Dendrogram tab to choose this visually — cut where vertical lines are longest.
- Linkage: How distance between clusters is measured as they merge. Ward is the best general-purpose choice.
- Silhouette Score: How similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good.
- Try different linkage methods and compare Silhouette Scores to find the best configuration.

**Visualizations:** Dendrogram, Scatter Plot
- Dendrogram: Each merge represents two clusters combining. The red dashed line shows where the dendrogram is cut for your chosen k. Longer vertical lines before the cut = more distinct clusters.
- Scatter Plot: Each color represents a cluster. Data is projected to 2D for visualization, and axis labels show how much variance each direction captures.


### PCA (Principal Component Analysis)

| Hyperparameter | What it does |
|----------------|-------------|
| **Number of components** | How many principal components to use for the process. |

**Output metrics:** Components, Variance Explained, PC1 Variance
- Components: The number of principal components used, each capturing a different direction of variance.
- Variance Explained: Cumulative percentage of the dataset's variance captured by all used components. 80% to 90% and above is generally good.
- PC1 Variance: The percent of variance captured by the first component alone.
- Adjust the number of components and watch the cumulative variance. Ideally, have high variance explained with few components.

**Visualizations:** Scatter Plot, Cumulative Variance
- Scatter Plot: Each point is one observation projected onto the two strongest principal components. Points that cluster together are similar across the selected features. The axis labels show how much variance each component captures.
- Cumulative Variance: How much total variance is captured by the model with each additional component. The colored lines mark the 80% and 90% thresholds, which are common targets/reference points for retaining enough information.


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

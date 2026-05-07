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


## Models

### K-Means Clustering

| Hyperparameter | What it does |
|----------------|-------------|
| **k (n_clusters)** | The number of clusters to form. Use the Elbow Plot tab to help choose the best k. |
| **Initialization method** | k-means++: smart initialization that speeds up convergence and avoids poor results. random: picks random starting centroids. |
| **n_init** | How many times the algorithm runs with different random seeds. The best result is kept. Higher = more stable but slower. |
| **Max iterations** | Maximum number of iterations per run before stopping. Increase if the model is not converging. |
| **Random seed** | Controls random initialization. Change this to test stability. |

**Output metrics:** Clusters, Inertia (Sum of Squared Distance), Silhouette Score
- Inertia: Sum of squared distances from each point to its cluster center. Lower = tighter, more compact clusters. Compare across different values of k.
- Silhouette Score: Measures how similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good.

**Visualizations:** Elbow Plot, Scatter Plot, Silhouette Plot
- Elbow Plot: Look for the 'elbow', the point where inertia starts decreasing more slowly. The red dashed line marks your current k. Try adjusting k to where the curve bends.
- Scatter Plot: Each color represents a cluster. X marks show the cluster centroids. Data is projected onto 2 principal components for visualization, and the axis labels show how much variance each direction captures.
- Silhouette Plot: Each colored band is a cluster. Wider bands mean more data points in that cluster. Bands that extend past the red average line indicate well-separated clusters. Thin or negative bands suggest that cluster may overlap with another.

---

### Hierarchical Clustering

| Hyperparameter | What it does |
|----------------|-------------|
| **k (n_clusters)** | Where to 'cut' the dendrogram. The Dendrogram tab helps you choose this visually. Cut where the vertical lines are longest. |
| **Linkage method** | ward: minimizes variance within clusters. Best general-purpose choice. complete: uses the maximum distance between cluster members. average: uses the mean distance between all pairs. single: uses the minimum distance, which is prone to chaining. |
| **Distance metric** | How distance between points is measured. Euclidean is the standard choice. (Ward linkage requires Euclidean.) |

**Output metrics:** Clusters (k), Linkage, Silhouette Score
- Silhouette Score: Measures how similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good.

**Visualizations:** Dendrogram, Cluster Scatter, Silhouette Analysis
- Dendrogram: Each merge represents two clusters combining. The red dashed line shows where the dendrogram is cut for your chosen k. Longer vertical lines before the cut = more distinct clusters.
- Cluster Scatter: Each color represents a cluster. Data is projected to 2D for visualization, and axis labels show how much variance each direction captures.
- Silhouette Analysis: Wider bands = more data points in that cluster. Bands past the red average line = well-separated clusters. Thin or negative bands suggest that cluster overlaps with another.

---

### PCA (Principal Component Analysis)

| Hyperparameter | What it does |
|----------------|-------------|
| **n_components** | How many principal components to retain. More components = more variance explained. Use the Cumulative Variance tab to find the right number. |
| **Random seed** | Controls random initialization. Change this to test stability. |

**Output metrics:** Components, Variance Explained, PC1 Variance
- Variance Explained: Cumulative % of the original dataset's variance captured by all retained components. 80–90%+ is generally considered good.
- PC1 Variance: % of variance captured by the first (strongest) principal component alone. A very high value means one direction dominates the data.

**Visualizations:** Scree Plot, Cumulative Variance, Component Loadings
- Scree Plot: Each bar shows how much variance a single principal component captures. Look for where the bars level off, so components after that point add little new information.
- Cumulative Variance: Shows how much total variance is captured as you add more components. The orange and red lines mark the 80% and 90% thresholds, which are common targets for retaining enough information.
- Component Loadings: Each cell shows how strongly a feature contributes to a principal component. Red = strong positive loading, blue = strong negative loading. Features with high absolute values drive that component.


## References

For more information or clarification about some of the machine learning topics used, here are some references I used:
- [What is K-Means Clustering?](https://www.youtube.com/watch?v=4b5d3muPQmA&t=18s)
- [What is Hierarchical Clustering?](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [What is PCA and how is it used for dimensionality reduction?](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [What is an Elbow Plot?](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)
- [What is a Dendrogram??](https://www.geeksforgeeks.org/scipy-cluster-hierarchy-dendrogram/)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

## PAGE SET UP
st.set_page_config(page_title="Unsupervised Machine Learning Tool", layout="wide")
st.title("Unsupervised Machine Learning Tool")
st.markdown("Upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance.")


# Caching here to prevent heavy computations from rerunning on every widget change, had some issues with the app being slow or crashing

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def prepare_data(feature_cols, data):
    """Encode and scale the selected features. Cached by column selection + data."""
    subset = data[list(feature_cols)].copy()
    
    for col in subset.columns:
        if not pd.api.types.is_numeric_dtype(subset[col]):
            subset[col] = OrdinalEncoder().fit_transform(subset[[col]])
            
    subset  = subset.dropna()
    X       = subset.values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, subset, len(data) - len(subset)

@st.cache_data
def elbow_inertias(X_scaled, init, n_init, max_iter):
    """Run KMeans for k=1..15 and return WCSS. Cached so a change in the slider doesn't have to rerun this."""
    inertias = []
    
    for k in range(1, 16):
        km = KMeans(n_clusters=k, init=init, n_init=n_init,
                    max_iter=max_iter)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


@st.cache_data
def cached_linkage(X_scaled, method):
    """Compute linkage matrix for dendrogram. Cached so it only reruns when data/method changes."""
    return linkage(X_scaled, method=method)



## SIDEBAR
with st.sidebar:

    st.header("Upload Dataset")
    dataset = st.file_uploader("Upload a CSV file", type="csv")
    df = None
    
    # Ask to upload a CSV file and read it once it is uploaded
    if dataset:
        df = load_csv(dataset)
    else:
        st.info("Upload a CSV file.")

    
        # Once the dataset is uploaded, reveal the widgets
    if df is not None:
        st.divider()
        st.header("Choose Variables")
        all_cols     = df.columns.tolist()
        
        # Allow picking as many explanatory variables as you want, as long as it isn't already picked
        feature_cols = st.multiselect(
            "Feature variables (columns to include in analysis)",

            #default to have 2 immediately show up so it runs right away
            all_cols, default=all_cols[:2])
        if not feature_cols:
            st.error("Select one or more feature variables.")
            st.stop()

        st.divider()

        # Choose from the models and have a help section so the user knows exactly what model they are picking.
        st.header("Choose a Model")
        model_name = st.selectbox(
            "Model",
            ["K-Means Clustering", "Hierarchical Clustering", "PCA"],
            help=("K-Means: partitions data into k clusters by minimizing within cluster distance.\n"
                  "Hierarchical: builds a tree of clusters, no need to pre-specify k upfront.\n"
                  "PCA: reduces dimensions while preserving as much variance as possible."))

        st.divider()
        st.header("Tune Hyperparameters")
        st.caption("Tune for model testing and performance.")

        model_params: dict = {}

        if model_name == "K-Means Clustering":
            model_params["n_clusters"] = st.slider(
                "Number of clusters", 2, 15, 3,
                help="The number of clusters to form. Use the Elbow Plot tab to choose the best k.")
            model_params["n_init"] = st.slider(
                "Number of initializations", 1, 20, 10,
                help="How many times the algorithm runs with different seeds. The best result is kept.")
            model_params["max_iter"] = st.slider(
                "Max iterations", 50, 500, 300, step=50,
                help="Maximum iterations per run. Increase if the model is not converging.")

        elif model_name == "Hierarchical Clustering":
            model_params["n_clusters"] = st.slider(
                "Number of clusters", 2, 15, 3,
                help="Where to 'cut' the dendrogram. The Dendrogram tab helps you choose this visually.")
            model_params["linkage"] = st.selectbox(
                "Linkage method", ["ward", "complete", "average", "single"],
                help=("ward: minimizes variance within clusters. Best general purpose choice.\n"
                      "complete: uses the maximum distance between cluster members.\n"
                      "average: uses the mean distance between all pairs.\n"
                      "single: uses the minimum distance, which is prone to chaining."))
            model_params["metric"] = "euclidean"


        elif model_name == "PCA":
            numeric_cols   = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            max_components = max(len(numeric_cols), 2)
            model_params["n_components"] = st.slider(
                "Number of components", 1, max_components, min(2, max_components),
                help="How many principal components to retain. Use the Cumulative Variance tab to choose.")

## MAIN PANEL
if df is None:
    st.info("Upload a dataset to perform analysis.")
    st.stop()
    
# Dataset preview
with st.expander("Quick Dataset Preview", expanded=True):
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("**Column Names:**")
        # Manually build dtype table as plain strings since passing raw dtype objects to st.dataframe crashes pyarrow in newer numpy versions.

        # Left column will show the column names and their data type
        
        dtype_df = pd.DataFrame({
            "column": df.columns.tolist(),
            "type":   [str(dt) for dt in df.dtypes]
        })
        st.dataframe(dtype_df, width="stretch", hide_index=True)
        
        #Underneath, still in the left column, are the dimensions of the dataset
        st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

    with right_col:
        st.markdown("**First 12 rows:**")
        # Right column will generate the first 12 rows
        st.dataframe(df.head(12), width="stretch")

    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe(), width="stretch")

## TRAINING

    # Choose st.spinner to have the model trained right as a choice/change is made on the sidebar
with st.spinner("Running the model..."):
    try:
        # prepare_data is cached, only reruns when feature selection or data changes
        X_scaled, all_var, num_dropped = prepare_data(tuple(feature_cols), df)

        if num_dropped == 1:
            st.caption("1 row with missing values was dropped before training.")
        elif num_dropped > 1:
            st.caption(f"{num_dropped} rows with missing values were dropped before training.")

        ## K-MEANS CLUSTERING
        if model_name == "K-Means Clustering":

            model  = KMeans(**model_params)
            labels = model.fit_predict(X_scaled)

            inertia   = model.inertia_
            sil_score = silhouette_score(X_scaled, labels)

            st.subheader("Model Performance: K-Means Clustering")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters (k)",     model_params["n_clusters"])
            col2.metric("WCSS",          f"{inertia:.2f}")
            col3.metric("Silhouette Score", f"{sil_score:.4f}")

            # WHAT DOES A GOOD SCORE LOOK LIKE?
            # This will be a dropdown expander below the presented metrics that explains what each is and a good score for them.
            
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Clusters (k)** | The number of clusters you chose. Use the Elbow Plot to find the best k. |
| **WCSS** | Sum of squared distances from each point to its cluster center. Lower = tighter clusters. Compare across different values of k. |
| **Silhouette Score** | How similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good. |
Adjust k and watch both metrics together to find the best number of clusters.""")

            tab1, tab2, tab3 = st.tabs(["Elbow Plot", "Silhouette Plot", "Scatter Plot"])


                # Elbow plot
            with tab1:
                # elbow_inertias is cached, so changing k slider won't rerun all 15 fits

                inertias = elbow_inertias(
                    X_scaled,
                    "k-means++",
                    model_params["n_init"],
                    model_params["max_iter"])

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, 16), inertias, marker="o", color="blue", lw=2)
                ax.axvline(model_params["n_clusters"], color="red", linestyle="--", lw=1.5,
                           label=f"Current k = {model_params['n_clusters']}")
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("WCSS")
                ax.set_title("Elbow Plot")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Look for the 'elbow' aka the point where inertia starts decreasing more slowly. The red dashed line marks your current k. Try adjusting k to where the curve bends.")

            # Scatter plot with clusters shown
            with tab3:
                pca_2d  = PCA(n_components=2)
                X_2d    = pca_2d.fit_transform(X_scaled)
                var_exp = pca_2d.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                                     alpha=0.7, edgecolors="white", s=50)
                centers_2d = pca_2d.transform(model.cluster_centers_)
                ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                           c="black", marker="X", s=50, zorder=5, label="Centroids")
                ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
                ax.set_title("Cluster Scatter (PCA-reduced to 2D)")
                ax.legend()
                plt.colorbar(scatter, ax=ax, label="Cluster")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each color represents a cluster. X marks show centroids. Data is projected to 2D for visualization, and axis labels show how much variance each direction captures.")

            # Silhouette plot
            with tab2:
                sil_vals   = silhouette_samples(X_scaled, labels)
                n_clusters = model_params["n_clusters"]
                colors     = plt.cm.tab10(np.linspace(0, 1, n_clusters))

                fig, ax = plt.subplots(figsize=(6, max(4, n_clusters * 0.8)))
                y_lower = 10
                for i in range(n_clusters):
                    vals    = np.sort(sil_vals[labels == i])
                    size    = vals.shape[0]
                    y_upper = y_lower + size
                    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
                    ax.text(-0.05, y_lower + 0.5 * size, str(i))
                    y_lower = y_upper + 10

                ax.axvline(sil_score, color="red", linestyle="--", lw=1.5,
                           label=f"Avg = {sil_score:.4f}")
                ax.set_xlabel("Silhouette coefficient")
                ax.set_ylabel("Cluster")
                ax.set_title("Silhouette Analysis by Cluster")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Wider bands = more data points in that cluster. Bands past the red average line = well-separated clusters. Thin or negative bands suggest that cluster overlaps with another.")

        ## HIERARCHICAL CLUSTERING
        elif model_name == "Hierarchical Clustering":

            model  = AgglomerativeClustering(**model_params)
            labels = model.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)

            st.subheader("Model Performance: Hierarchical Clustering")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters (k)",     model_params["n_clusters"])
            col2.metric("Linkage",          model_params["linkage"].capitalize())
            col3.metric("Silhouette Score", f"{sil_score:.4f}")

            # WHAT DOES A GOOD SCORE LOOK LIKE?
            # This will be a dropdown expander below the presented metrics that explains what each is and a good score for them.
            
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Clusters (k)** | Where the dendrogram is "cut." Use the Dendrogram tab to choose this visually, cut where vertical lines are longest. |
| **Linkage** | How distance between clusters is measured as they merge. Ward is the best general-purpose choice. |
| **Silhouette Score** | How similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Above 0.5 is good. |
Try different linkage methods and compare Silhouette Scores to find the best configuration.""")

            tab1, tab3 = st.tabs(["Dendrogram", "Scatter Plot"])

            # Dendrogram
            with tab1:
                # cached_linkage only reruns when data or linkage method changes
                Z = cached_linkage(X_scaled, model_params["linkage"])

                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, truncate_mode="lastp", p=50,
                           leaf_rotation=90, leaf_font_size=8, ax=ax, color_threshold=0)
                cut_height = Z[-(model_params["n_clusters"] - 1), 2]
                ax.axhline(cut_height, color="red", linestyle="--", lw=1.5,
                           label=f"Cut for k = {model_params['n_clusters']}")
                ax.set_xlabel("Data points")
                ax.set_ylabel("Distance")
                ax.set_title(f"Dendrogram: {model_params['linkage'].capitalize()} Linkage")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each merge represents two clusters combining. The red dashed line shows where the dendrogram is cut for your chosen k. Longer vertical lines before the cut = more distinct clusters.")

            # Scatter plot
            with tab3:
                pca_2d  = PCA(n_components=2)
                X_2d    = pca_2d.fit_transform(X_scaled)
                var_exp = pca_2d.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                                     alpha=0.7, edgecolors="white", s=50)
                ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
                ax.set_title("Cluster Scatter (PCA reduced to 2D)")
                plt.colorbar(scatter, ax=ax, label="Cluster")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Each color represents a cluster. Data is projected to 2D for visualization, and axis labels show how much variance each direction captures.")


        ## PCA
        elif model_name == "PCA":

            numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_features) < 2:
                st.error("PCA requires at least 2 numeric feature columns. Please select more numeric columns.")
                st.stop()

            X_pca        = all_var[numeric_features].values.astype(float)
            X_pca_scaled = StandardScaler().fit_transform(X_pca)

            model         = PCA(**model_params)
            X_transformed = model.fit_transform(X_pca_scaled)

            evr            = model.explained_variance_ratio_
            cumulative_var = np.cumsum(evr)
            n_components   = model_params["n_components"]

            st.subheader("Model Performance: PCA")
            col1, col2, col3 = st.columns(3)
            col1.metric("Components",         n_components)
            col2.metric("Variance Explained", f"{cumulative_var[-1]*100:.2f}%")
            col3.metric("PC1 Variance",       f"{evr[0]*100:.2f}%")


            # WHAT DOES A GOOD SCORE LOOK LIKE?
            # This will be a dropdown expander below the presented metrics that explains what each is and a good score for them.
            
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Components** | The number of principal components retained. Each captures a different direction of variance. |
| **Variance Explained** | Cumulative % of the dataset's variance captured by all retained components. 80–90%+ is generally good. |
| **PC1 Variance** | % of variance captured by the first component alone. A very high value means one direction dominates the data. |
Adjust the number of components and watch the cumulative variance. Aim for the fewest components that still explain 80–90% of variance.""")

            tab1, tab2 = st.tabs(["Scatter Plot", "Cumulative Variance"])

                # Scatter Plot
            with tab1:
                if n_components >= 2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                                         alpha=0.7, edgecolors="white", s=50, color="blue")
                    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
                    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
                    ax.set_title("PCA Scatter Plot (PC1 vs PC2)")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("Each point is one observation projected onto the two strongest principal components. Points that cluster together are similar across the selected features. The axis labels show how much variance each component captures.")
                else:
                    st.info("Select at least 2 components to display the scatter plot.")
                    
            #Cumulative Variance
            with tab2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, n_components + 1), cumulative_var * 100,
                        marker="o", color="blue", lw=2)
                ax.axhline(80, color="orange", linestyle="--", lw=1, label="80% threshold")
                ax.axhline(90, color="red",    linestyle="--", lw=1, label="90% threshold")
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Variance Explained (%)")
                ax.set_title("Cumulative Explained Variance")
                ax.set_ylim(0, 105)
                ax.set_xticks(range(1, n_components + 1))
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Shows how much total variance is captured as you add more components. The orange and red lines mark the 80% and 90% thresholds, which are common targets/reference points for retaining enough information.")

    except Exception as e:
        st.error(f"Model failed: {e}")
        st.exception(e)

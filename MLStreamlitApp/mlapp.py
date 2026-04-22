import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
)

## Page setup
st.set_page_config(
    page_title="Supervised Machine Learning Tool",
    layout="wide")

st.title("Supervised Machine Learning Tool")
st.markdown(
    "Upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance.")


## Sidebar
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")

    df = None
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.info("Upload a CSV file")

    if df is not None:
        st.divider()
        st.header("Choose Variables")
        target_col = st.selectbox(
            "Variable to estimate",
            df.columns.tolist(),
            index=len(df.columns) - 1,
        )
        feature_cols = st.multiselect(
            "Features",
            [c for c in df.columns if c != target_col],
            default=[c for c in df.columns if c != target_col],
        )

        st.divider()
        st.header("Choose a Model")
        model_name = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Logistic Regression"],
            help=(
                "Linear Regression → continuous numeric target  \n"
                "Logistic Regression → categorical / binary target"
            ),
        )

        st.divider()
        st.header("Tune Hyperparameters")

        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=100, step=1)

        model_params: dict = {}

        if model_name == "Linear Regression":
            model_params["alpha"] = st.slider(
                "Regularization strength (α)",
                0.0, 10.0, 1.0, 0.1,
                help="Higher α → stronger regularization, shrinks coefficients toward zero. α = 0 is equivalent to plain OLS.",
            )

        elif model_name == "Logistic Regression":
            model_params["C"] = st.slider(
                "C (inverse regularization strength)",
                0.01, 10.0, 1.0, 0.01,
                help="Smaller C → stronger regularization.",
            )
            model_params["penalty"] = st.selectbox(
                "Penalty",
                ["l2", "l1"],
                help="L2 shrinks all coefficients toward zero; L1 can zero out coefficients entirely (built-in feature selection).",
            )
            # liblinear is used because it supports both l1 and l2
            model_params["solver"] = "liblinear"
            model_params["max_iter"] = 1000
            model_params["random_state"] = int(random_state)

        scale_features = st.checkbox("Scale features (StandardScaler)", value=True)

        st.divider()
        train_btn = st.button("Train Model", use_container_width=True, type="primary")

## Main panel
if df is None:
    st.info("Upload a dataset")
    st.stop()

# Data preview
with st.expander("Quick view of Dataset", expanded=True):
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(11), use_container_width=True)
    st.write("**Descriptive statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

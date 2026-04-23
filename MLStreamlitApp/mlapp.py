import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score)



## PAGE SET UP

st.set_page_config(page_title = "Supervised Machine Learning Tool", layout = "wide")
st.title("Supervised Machine Learning Tool")
st.markdown("Upload a dataset, experiment with hyperparameters, and observe how these affect model training and performance.")


## SIDEBAR

with st.sidebar:
    
    st.header("Upload Dataset")
    
    dataset = st.file_uploader("Upload a CSV file", type = "csv")

    df = None

        # Ask to upload a CSV file and read it once it is uploaded
    if dataset:
        df = pd.read_csv(dataset)
    else:
        st.info("Upload a CSV file.")

    
        # Once the dataset is uploaded, reveal the widgets
    if df is not None:
        st.divider()
        st.header("Choose Variables")

            # Have a select box that allows for ONE variable to be predicted
        target_col = st.selectbox(
            "Target variable (what you want to predict)",
            df.columns.tolist(),
            index = len(df.columns) - 1)

            # Allow picking as many explanatory variables as you want, as long as it isn't the target
        non_targets = [c for c in df.columns if c != target_col]
        feature_cols = st.multiselect(
            "Feature variables (what you are using to predict)",
            non_targets)

        

        st.divider()
        st.header("Choose a Model")
        model_name = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Logistic Regression"],
            help=(
                "Linear Regression for continuous numeric target (e.g. price, score)\n"
                "Logistic Regression for binary categorical target (e.g. yes/no, win/loss, gender)"))


        st.divider()
        st.header("Tune Hyperparameters")
        st.caption("Results update live as you adjust these.")

        random_state = st.number_input("Random seed", value=100, step=1,
            help="Controls the train/test split. Change this to test stability.")
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05,
            help="Fraction of data held out for evaluation. 0.2 = 20% test, 80% train.")



        model_params: dict = {}

        if model_name == "Linear Regression":
            
            model_params["alpha"] = st.slider(
                "Regularization strength (α)",
                0.0, 10.0, 1.0, 0.01,
                help = "α penalizes larger coefficients, reducing overfitting. The higher the α, the stronger the shrinkage. α = 0 is regular OLS.")
            
            model_params["penalty"] = st.selectbox(
                "Penalty type",
                ["L2", "L1"],
                help = "L2 (Ridge): shrinks coefficients to zero.\nL1 (Lasso): zeros out coefficients entirely for accuracy and interpretability.")

        elif model_name == "Logistic Regression":
            
            model_params["C"] = st.slider(
                "Inverse regularization strength (C)",
                0.01, 10.0, 1.0, 0.01,
                help = "Balances keeping coefficients small and fitting training data.")
            
            model_params["penalty"] = st.selectbox(
                "Penalty type",
                ["L2", "L1"],
                help = "L2 (Ridge): shrinks coefficients to zero.\nL1 (Lasso): zeros out coefficients entirely for accuracy and interpretability.")
            
            # liblinear supports both L1 and L2 and works well on small-to-medium datasets
            model_params["solver"] = "liblinear"
            model_params["max_iter"] = 1000
            model_params["random_state"] = int(random_state)



## MAIN PANEL
if df is None:
    st.info("Upload a dataset to perform analysis.")
    st.stop()

# Dataset preview
with st.expander("Quick Dataset Preview", expanded = True):
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
            # Left column will show the column names and their data type
        st.markdown("**Column Names:**")
        st.dataframe(df.dtypes.rename("type").reset_index().rename(columns={"index": "column"}),
                     use_container_width = True, hide_index = True)
        
            #Underneath, still in the left column, are the dimensions of the dataset
        st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

    
    with right_col:
            # Right column will generate the first 12 rows
        st.markdown("**First 12 rows:**")
        st.dataframe(df.head(12), use_container_width = True)

        # Under both columns will be the descriptive statistics
    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe(), use_container_width = True)
    

# Verify that the model matches the data type of the target variable (or else model will not work)

    # Make all continuous (for later)
def is_continuous(series):
    """A numeric column with more than 10 unique values is treated as continuous."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10

target_is_continuous = is_continuous(df[target_col])

if model_name == "Linear Regression" and not target_is_continuous:
    st.error(
        f"**Improper model** '{target_col}' is not numerical."
        f"({df[target_col].nunique()} unique values). "
        "**Linear Regression requires a continuous numeric target.** "
        "Switch to Logistic Regression if binary or choose a different target column.")
    st.stop()

if model_name == "Logistic Regression" and target_is_continuous:
    st.error(
        f"**Model mismatch:** '{target_col}' is not binary."
        f"({df[target_col].nunique()} unique values). "
        "**Logistic Regression requires a binary categorical target.** "
        "Switch to Linear Regression if continuous numerical or choose a different target column.")
    st.stop()


## ADAPTIVE TRAINING


with st.spinner("Training the model"):
    
    try:
        working = df[feature_cols + [target_col]].copy()

        # For logistic regression, save original label names before any encoding
        original_target_labels = None
        if model_name == "Logistic Regression":
            le = LabelEncoder()
            le.fit(working[target_col].astype(str))
            original_target_labels = le.classes_          # e.g. ['Female', 'Male']
            working[target_col] = le.transform(working[target_col].astype(str))

        # Encode any remaining text columns in features
        for col in working[feature_cols].select_dtypes(include=["object", "category"]).columns:
            working[col] = OrdinalEncoder().fit_transform(working[[col]])

        # Drop rows with missing values
        before = len(working)
        working = working.dropna()
        dropped = before - len(working)
        if dropped > 0:
            st.warning(f"⚠️ {dropped} row(s) with missing values were dropped before training.")

        X = working[feature_cols].values
        y = working[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=(y if model_name == "Logistic Regression" else None),
        )

        # Scale features — important for regularized models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        

        ## LINEAR REGRESSION SECTION
        
        if model_name == "Linear Regression":
            penalty = model_params.pop("penalty")  # Ridge = L2, LASSO = L1
            LinearModel = Lasso if penalty == "l1" else Ridge
            model = LinearModel(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2   = r2_score(y_test, y_pred)
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_test, y_pred)

            st.subheader("Model Performance: Linear Regression")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2:.5f}")
            col2.metric("MSE", f"{mse:.5f}")
            col3.metric("RMSE", f"{rmse:.5f}")
            col4.metric("MAE", f"{mae:.5f}")


            # WHAT DOES A GOOD SCORE LOOK LIKE?
            
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **R²** | The proportion of variance in the prediction variable explained by the model. 1.0 means a perfect model, 0 means no better than always predicting the sample mean. 0.7+ is good, 0.9+ is excellent |
| **MSE** | Mean squared error of predictions. Larger errors are penalized more. The lower the score, the better. Compare this across different hyperparameters |
| **RMSE** | Square root of MSE. Same units as the target, easier to interpret. Ideally, small compared to the range of the predicted variable.|
| **MAE** | Average absolute error of predictions. Less sensitive to outliers and large errors than RMSE. |
    **Adjust hyperparameter α to improve R².**""")


            table1, table2, table3 = st.tabs(
                ["Predicted vs Actual", "Residuals", "Feature Coefficients"]
            )

            with table1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred, alpha=0.6, color="#185FA5", edgecolors="white", s=50)
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
                ax.set_xlabel("Actual values")
                ax.set_ylabel("Predicted values")
                ax.set_title("Predicted vs Actual")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Points closer to the dashed line indicate accurate predictions, and distance between points and line shows prediction error (residuals).")

            with table2:
                residuals = y_test - y_pred
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].scatter(y_pred, residuals, alpha=0.6, color="blue", edgecolors="white", s=50)
                axes[0].axhline(0, color="black", lw=1.5, linestyle="--")
                axes[0].set_xlabel("Predicted values")
                axes[0].set_ylabel("Residuals")
                axes[0].set_title("Residuals vs Fitted")
                axes[1].hist(residuals, bins=20, color="purple", edgecolor="white")
                axes[1].set_xlabel("Residual")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Residual Distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("In an ideal model, the residuals are randomly scattered around zero and in an approximate bell curve distribution.")

            with table3:
                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": model.coef_
                }).sort_values("Coefficient", key=abs, ascending=True)
                fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.35)))
                colors = ["red" if c < 0 else "green" for c in coef_df["Coefficient"]]
                ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
                ax.axvline(0, color="black", lw=1)
                ax.set_xlabel("Coefficient value")
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green bars show a positive relationship with the target, red bars show a negative relationship. Longer bars mean larger coefficients in magnitude.")

        ## LOGISTIC REGRESSION SECTION
        
        
        elif model_name == "Logistic Regression":
            model_params["penalty"] = model_params["penalty"].lower()
            model = LogisticRegression(**model_params)
            model = LogisticRegression(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            class_indices  = np.unique(y).astype(int)
            display_labels = (original_target_labels[class_indices]
                              if original_target_labels is not None
                              else class_indices.astype(str))
            n_classes = len(class_indices)
            is_binary = n_classes == 2
            avg = "binary" if is_binary else "weighted"

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

            y_prob = model.predict_proba(X_test)
            auc = (roc_auc_score(y_test, y_prob[:, 1])
                   if is_binary
                   else roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))

            st.subheader("Model Performance: Logistic Regression")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy",  f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall",    f"{rec:.4f}")
            col4.metric("F1 Score",  f"{f1:.4f}")
            col5.metric("AUC-ROC",   f"{auc:.4f}")



            # WHAT DOES A GOOD SCORE LOOK LIKE?
            with st.expander("How to interpret these metrics"):
                st.markdown("""
| Metric | What it measures | 
|--------|-----------------|
| **Accuracy** | Percentage of all predictions that are correct. Can be misleading on imbalanced datasets. 0.80+ is good|
| **Precision** | Of all predicted positives, how many were actually positive? High precision = few false alarms. 0.80+ is good |
| **Recall** | Of all actual positives, how many did the model catch? High recall = few missed cases. 0.80+ is good |
| **F1 Score** | Harmonic mean of Precision and Recall. Best single metric when classes are imbalanced. 0.80+ is good|
| **AUC-ROC** | Area under the ROC curve. Measures how well the model separates classes regardless of threshold. 0.5 means no better than random guessing, 0.80+ is good|

        **Adjust C to find the right balance out the differing effects of false negatives and false positives.**""")

            table1, table2, table3 = st.tabs(
                ["Confusion Matrix", "ROC Curve", "Feature Coefficients"])

            with table1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=display_labels, yticklabels=display_labels, ax=ax
                )
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Top left are true negatives and bottom right are true positives, these are correct predictions. Top right are false positives and bottom left are false negatives, these are the errors.")

            with table2:
                fig, ax = plt.subplots(figsize=(5, 4))
                if is_binary:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="blue", lw=2)
                else:
                    for i, (idx, lbl) in enumerate(zip(class_indices, display_labels)):
                        fpr, tpr, _ = roc_curve((y_test == idx).astype(int), y_prob[:, i])
                        ax.plot(fpr, tpr, lw=1.5, label=str(lbl))
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Shows the tradeoff between true positives and false positive rates. A convex curve bending towards the top left corner is ideal. A straight, linear line represents an AUC of 0.5.")

            with table3:
                coefs = (np.abs(model.coef_).mean(axis=0)
                         if model.coef_.ndim > 1
                         else model.coef_[0])
                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": coefs,
                }).sort_values("Coefficient", key=abs, ascending=True)
                fig, ax = plt.subplots(figsize=(5, max(3, len(feature_cols) * 0.35)))
                colors = ["red" if c < 0 else "green" for c in coef_df["Coefficient"]]
                ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
                ax.axvline(0, color="black", lw=1)
                ax.set_xlabel("Coefficient value")
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green bars increase the log-odds of the positive class; red bars decrease them. Longer bars indicate stronger influence. Try switching between L1 and L2 penalty, L1 may zero out some bars entirely.")



    except Exception as e:
        st.error(f"Training failed: {e}")
        st.exception(e)

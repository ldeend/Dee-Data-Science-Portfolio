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
                # Default to have a random variable selected so the model works immediately
            non_targets, default = non_targets[:1])
        if not feature_cols:
            st.error("Select one or more feature variables")
            st.stop()
        

        st.divider()
        st.header("Choose a Model")

            # Choose from the two models and have a help section so the user knows exactly what model they are picking.
        model_name = st.selectbox(
            "Model",
            ["Linear Regression", "Logistic Regression"],
            help=("Linear Regression for continuous numeric target (e.g. price, score)\n"
                "Logistic Regression for binary categorical target (e.g. yes/no, win/loss, gender)"))


        st.divider()
        st.header("Tune Hyperparameters")
        st.caption("Tune for model testing and perfomance.")

            # Since the train test split comes from a random seed, allow the user to change the seed to get different outcomes to test the stability of the model
        random_state = st.number_input("Random seed", value=100, step=1,
            help="Controls the train/test split. Change this to test stability.")

            # Split should be set at 20% and 80% train, but allow the option to change if the user wants
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05,
            help="Fraction of data held out for evaluation. 0.2 = 20% test, 80% train.")


        # Model hyperparameter section. Each model has different hyperparameters taht are described in help sections if needed.
        
        model_params: dict = {}

        if model_name == "Linear Regression":
            
            model_params["alpha"] = st.slider(
                "Regularization strength (α)",
                    # Chose range 0 to 10 for alpha and C
                0.0, 10.0, 1.0, 0.01,
                help = "α penalizes larger coefficients, reducing overfitting. The higher the α, the stronger the shrinkage. α = 0 is regular OLS.")
            
            model_params["penalty"] = st.selectbox(
                "Penalty type",
                ["L2", "L1"],
                help = "L2 (Ridge): shrinks coefficients to zero.\nL1 (Lasso): zeros out coefficients entirely for accuracy and interpretability.")

        elif model_name == "Logistic Regression":
            
            model_params["C"] = st.slider(
                "Inverse regularization strength (C)",
                0.0, 10.0, 1.0, 0.01,
                help = "Balances keeping coefficients small and fitting training data.")
            
            model_params["penalty"] = st.selectbox(
                "Penalty type",
                ["L2", "L1"],
                help = "L2 (Ridge): shrinks coefficients to zero.\nL1 (Lasso): zeros out coefficients entirely for accuracy and interpretability.")
            
            # Can use liblinear or any other solver to solve L1 and L2 regularization problems
            model_params["solver"] = "liblinear"

            # Want high iterations to get a good result, but not too high that the computation takes long. Tested 1000 and works so will use.
            model_params["max_iter"] = 1000

            # Use the random seed the user selected
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

def numeric(series):
    return pd.api.types.is_numeric_dtype(series)
if model_name == "Linear Regression" and not numeric(df[target_col]):
    st.error(f"'{target_col}' is not numeric. Linear Regression requires a numeric target. Switch to Logistic Regression or choose a different target column.")
    st.stop()


def binary(series):
    return series.nunique() == 2
if model_name == "Logistic Regression" and not binary(df[target_col]):
    st.error(f"'{target_col}' is not binary ({df[target_col].nunique()} unique values). Logistic Regression requires a binary target. Switch to Linear Regression or choose a different target column.")
    st.stop()


## TRAINING

    # Choose st.spinner to have the model trained right as a choice/change is made on the sidebar
with st.spinner("Training the model"):
  
    try:
        # All the columns/variables used
        all_var = df[feature_cols + [target_col]].copy()

        
        # Before actual training, we need to do a few things. We need to remember the target names (if binary) for later, 
        # drop rows with any missing values, and create the training and testing sets based on the train/test split.

        # For logistic regression, save original label names before any encoding for easier interpretation at the visualizations
        # This makes it so the confusion matrix says "Female" and "Male" instead of 0 and 1, as an example

        # Saving the target labels for logistic regression (used youtube.com/watch?v=15uClAVV-rI for Encoder explanation)
        target_labels = None
        if model_name == "Logistic Regression":
            encoder = LabelEncoder()
            encoder.fit(all_var[target_col].astype(str))
            target_labels = encoder.classes_      
            all_var[target_col] = encoder.transform(all_var[target_col].astype(str))

        # Encode any remaining text columns in features for sklearn to work properly.
        for col in all_var[feature_cols].select_dtypes(include = ["object", "category"]).columns:
            all_var[col] = OrdinalEncoder().fit_transform(all_var[[col]])
        
        # Drop rows with missing values so the model doesn't run into issues. Missing data is a different machine learning problem.
        old_len = len(all_var)
        all_var = all_var.dropna()
        num_dropped = old_len - len(all_var)
        if num_dropped == 1:
            st.caption("1 row with missing values was dropped before training.")
        elif num_dropped > 1:
            st.caption(f"{num_dropped} rows with missing values were dropped before training.")
        

            # Split the dataset into the training set and testing set based on the new, cleaned dataset        
        X = all_var[feature_cols].values
        y = all_var[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = float(test_size),
            random_state = int(random_state),
            stratify=(y if model_name == "Logistic Regression" else None))

        # Scale features so the regularized models can work properly
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        

        ## LINEAR REGRESSION SECTION

        
        if model_name == "Linear Regression":

            # Train the model with the correct regularization and train/test split
            penalty = model_params.pop("penalty") 
                # Remove penalty since sklearn can't work with it, but LinearModel can
            LinearModel = Lasso if penalty == "L1" else Ridge
            model = LinearModel(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate the evaluation metrics
            r2   = r2_score(y_test, y_pred)
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_test, y_pred)

            # Present the Evaluation Metrics  below the dataset preview
            st.subheader("Model Performance: Linear Regression")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2:.5f}")
            col2.metric("MSE", f"{mse:.5f}")
            col3.metric("RMSE", f"{rmse:.5f}")
            col4.metric("MAE", f"{mae:.5f}")


            # WHAT DOES A GOOD SCORE LOOK LIKE?
            # This will be a dropdown expander below the presented metrics that explains what each is and a good score for them.
            
            with st.expander("How do I interpret these metrics?"):
                st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **R²** | The proportion of variance in the prediction variable explained by the model. 1.0 means a perfect model, 0 means no better than always predicting the sample mean. 0.7+ is good, 0.9+ is excellent |
| **MSE** | Mean squared error of predictions. Larger errors are penalized more. The lower the score, the better. Compare this across different hyperparameters |
| **RMSE** | Square root of MSE. Same units as the target, easier to interpret. Ideally, small compared to the range of the predicted variable.|
| **MAE** | Average absolute error of predictions. Less sensitive to outliers and large errors than RMSE. |
    Adjust hyperparameter α to improve R².""")



            # Along with the metrics, there will be 3 tables: Predicted vs Actual values, Residuals (Predicted - Actual), and the Coefficients Visualized
            
            table1, table2, table3 = st.tabs(
                ["Predicted vs Actual", "Residuals", "Feature Coefficients"])

                # Predicted vs Actual with a reference line to perfect prediction to quickly see how well the model did
            with table1:
                fig, ax = plt.subplots(figsize = (5, 4))
                ax.scatter(y_test, y_pred, alpha = 0.5, color = "blue", edgecolors = "white", s = 50)
                    # limit the graph to just show where the values are
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                    # Perfectly linear dashed line as the reference
                ax.plot(lims, lims, "k--", lw=1, label = "Perfect Prediction")
                ax.set_xlabel("Actual values")
                ax.set_ylabel("Predicted values")
                ax.set_title("Predicted vs Actual")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                    # Add a caption below to explain the viz
                st.caption("Points closer to the dashed line indicate accurate predictions, and distance between points and line shows prediction error (residuals).")

                # Residuals plot. They should be white noise around the origin in the left image and approximately normally distributed
                # in the right image to have confidence in the model.
            with table2:
                residuals = y_test - y_pred
                fig, axes = plt.subplots(1, 2, figsize = (10, 4))
                axes[0].scatter(y_pred, residuals, alpha = 0.5, color = "blue", edgecolors = "white", s = 40)
                axes[0].axhline(0, color = "black", lw = 1, linestyle="--")
                axes[0].set_xlabel("Predicted values")
                axes[0].set_ylabel("Residuals")
                axes[0].set_title("Residuals vs Fitted")

                    # Making the residuals distribution
                axes[1].hist(residuals, bins = 25, color = "purple", edgecolor = "white")
                axes[1].set_xlabel("Residual")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Residual Distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("In an ideal model, the residuals are randomly scattered around zero and in an approximate bell curve distribution.")

                # This image is of the coefficients visualized with their relative sizes. This is to tell the user what the 
                # coefficients are but also shows the affect of the L1 vs L2 norms on the coefficients.
            with table3:
                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": model.coef_
                }).sort_values("Coefficient", key=abs, ascending=True)
                fig, ax = plt.subplots(figsize = (5, max(3, len(feature_cols) * 0.35)))
                    # Make positive values green and negative values red just for appearance
                bar_colors = ["red" if c < 0 else "green" for c in coef_df["Coefficient"]]
                ax.barh(coef_df["Feature"], coef_df["Coefficient"], color = bar_colors)
                ax.axvline(0, color = "black", lw = 1)
                ax.set_xlabel("Coefficient value")
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green bars show a positive relationship with the target, red bars show a negative relationship. Longer bars mean larger coefficients in magnitude.")

        ## LOGISTIC REGRESSION SECTION

        elif model_name == "Logistic Regression":

                # Train the model with the correct regularization and train/test split
            model_params["penalty"] = model_params["penalty"].lower()
                # Needs to be lowercase: we call it L1 and L2 but Python needs it in l1 or l2
            model = LogisticRegression(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
            show_labels = target_labels if target_labels is not None else np.array(["0", "1"])

            # Calculate the evaluation metrics
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
                # For ROC curve
            auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


            # Present the metrics
            st.subheader("Model Performance: Logistic Regression")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy",  f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall",    f"{recall:.4f}")
            col4.metric("F1 Score",  f"{f1:.4f}")
            col5.metric("AUC-ROC",   f"{auc:.4f}")





            # WHAT DOES A GOOD SCORE LOOK LIKE?
            # This will be a dropdown expander below the presented metrics that explains what each is and a good score for them.
            
            with st.expander("How to interpret these metrics"):
                st.markdown("""
| Metric | What it measures | 
|--------|-----------------|
| **Accuracy** | Percentage of all predictions that are correct. Can be misleading on imbalanced datasets. 0.80+ is good|
| **Precision** | Of all predicted positives, how many were actually positive? High precision = few false alarms. 0.80+ is good |
| **Recall** | Of all actual positives, how many did the model catch? High recall = few missed cases. 0.80+ is good |
| **F1 Score** | Harmonic mean of Precision and Recall. Best single metric when classes are imbalanced. 0.80+ is good|
| **AUC-ROC** | Area under the ROC curve. Measures how well the model separates classes regardless of threshold. 0.5 means no better than random guessing, 0.80+ is good|

        Adjust C to find the right balance out the differing effects of false negatives and false positives.""")

            table1, table2, table3 = st.tabs(
                ["Confusion Matrix", "ROC Curve", "Feature Coefficients"])

                # Confusion matrix showing TN, TP, FN, FP
            with table1:
                cmatrix = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize = (5, 4))
                sns.heatmap(
                    cmatrix, annot = True, fmt = "d", cmap = "Blues",
                    xticklabels = show_labels, yticklabels = show_labels, ax = ax)
                
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Top left are true negatives and bottom right are true positives, these are correct predictions. Top right are false positives and bottom left are false negatives, these are the errors.")

                # AUC curve
            with table2:
                
                fig, ax = plt.subplots(figsize = (5, 4))
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                ax.plot(fpr, tpr, label=f"AUC = {auc:.5f}", color = "blue", lw = 1.5)
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc = "lower right")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Shows the tradeoff between true positives and false positive rates. A convex curve bending towards the top left corner is ideal. A straight, linear line represents an AUC of 0.5.")

                # Another coefficients visualization
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


        # Have a neat error message if for some reason the training does not work instead of an ugly, long error message
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.exception(e)

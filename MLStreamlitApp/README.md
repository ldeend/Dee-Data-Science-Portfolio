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

### Dataset Upload and View
- Upload a CSV file through the Upload/Drop on the sidebar
- In the main panel, a preview of the first 10 rows, column names and types, and descriptive statistics
- Rows with missing values are dropped before the model is trained on it, and the app states how many were removed
- Text columns are encoded automatically using ordinal encoding, and everything is scaled properly (for regularization)

### Model Selection
- Select one target (Y) variable and as many feature (X) variables 
- Choose between Linear Regression for continuous target variables and Logistic Regression for binary target variables
- The app will notify if the model choice doesn't match the target variable
- You can choose a value for the hyperparameters, type of regularization, train/test split (though it should stay around 80% train), and try different random seeds

### Output
- Each model outputs variance evaluation metrics and visualizations to understand the model's performance
- For each model output, there is a dropdown panel titled **"How to interpret these metrics"** that explains what each metric is and what a good value is for each
- Along with each visualization is a short description
- Every time an aspect of the model is changed (tweaking a hyperparameter, adding a feature, etc.), it re-renders the model automatically, allowing for quick comparison and tuning 


## Breakdown by Model

### Linear Regression

| Hyperparameter | What it does |
|----------------|-------------|
| **α** | Penalizes larger coefficients, reducing overfitting. The higher the α, the stronger the shrinkage. α = 0 is regular OLS. |
| **Test split size** | Fraction of data held out for testing after training the model on the other part of the dataset. |
| **Random seed** | Different seeds have different splits, so can change it to test result stability. |

Can choose L2 Regularization (Ridge Regression) or L1 Regularization (LASSO)

**Output metrics:** R², MSE, RMSE, MAE  
- R² is the main one, 0.7+ is good and 0.9+ is amazing
- RMSE and MAE should be small relative to the range of your target variable

**Visualizations:** Predicted vs Actual scatter plot, Residuals vs Fitted, Residual distribution, Feature Coefficients Visualized
- Predicted vs Actual: Points closer to the dashed line indicate accurate predictions, and distance between points and line shows prediction error (residuals).
- Residual Analysis: In an ideal model, the residuals are randomly scattered around zero and in an approximate bell curve distribution.

### Logistic Regression

| Hyperparameter | What it does |
|----------------|-------------|
| **C** | Inverse of regularization strength. Balances keeping coefficients small and fitting training data. Larger c fits the training data more. |
| **Penalty** | L2 (Ridge): shrinks coefficients to zero. L1 (Lasso): zeros out coefficients entirely for accuracy and interpretability. |
| **Test set size** | Fraction of data held out for testing after training the model on the other part of the dataset. |
| **Random seed** | Different seeds have different splits, so can change it to test result stability. |

**Output metrics:** Accuracy, Precision, Recall, F1 Score, AUC-ROC  
- Accuracy: Percentage of all predictions that are correct. Can be misleading on imbalanced datasets. 0.80+ is good
- AUC-ROC: Area under the ROC curve. Measures how well the model separates classes regardless of threshold. 0.5 means no better than choosing at random. 0.80+ is good.
- F1 Score: Harmonic mean of Precision and Recall. Best single metric when classes are imbalanced. 0.80+ is good. One of the most relied on classification metrics.
  
**Visualizations:** Confusion Matrix, ROC Curve, Feature Coefficients Visualize
- Confusion Matrix: Top left are true negatives and bottom right are true positives, these are correct predictions. Top right are false positives and bottom left are false negatives, these are the errors.
- ROC Curve: Shows the tradeoff between true positives and false positive rates. A convex curve bending towards the top left corner is ideal. A straight, linear line represents an AUC of 0.5


## References

For more information or clarification about some of the machine learning topics used, here are the references I used:
- [What are different performance evaluation metrics and how are they used?](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [What is AUC and what is a ROC curve?](https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/)
- [What is L2 Regularization?](https://www.youtube.com/watch?v=Q81RR3yKn30) [What is L1 Regularization?](https://www.youtube.com/watch?v=NGf0voTMlcs)
- [What is Logistic Regression and how is it used differently than Linear Regression?](https://christophm.github.io/interpretable-ml-book/logistic.html)

## Example Screenshots

For these example screenshots, I used the men's college basketball data from 2013 from [this Kaggle link](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/data). 

**Example Dataset Preview:**
<img width="985" height="472" alt="Screenshot 2026-04-22 at 8 06 17 PM" src="https://github.com/user-attachments/assets/d50bf049-9798-4fc1-923c-7ea0cb0a8466" />

**Example Evaluation Metrics Panel:**
<img width="1049" height="566" alt="Screenshot 2026-04-22 at 8 22 17 PM" src="https://github.com/user-attachments/assets/2e7ec5fc-a466-405a-a51b-fa5fa8902363" />


**Example Performance Visualization:**
<img width="1460" height="573" alt="image" src="https://github.com/user-attachments/assets/41ae646b-c8c6-4b1b-ad9c-f43171d80a21" />



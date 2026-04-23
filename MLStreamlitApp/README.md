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



# Data Science Portfolio

This repository is intended to showcase my data science projects and capabilities. I will have work related to data analysis, modeling, and machine learning. Take a glance at my quick descriptions below and follow the links for any project that catches your interest.


## [College Basketball Stats Streamlit App](https://github.com/ldeend/Dee-Data-Science-Portfolio/tree/main/basic_streamlit_app)

### Project Overview
This is an interactive Streamlit app that allows users to look at and explore Division I Men's College Basketball teams and their stats via different filters. This project demonstrates skills in interactive data visualization, filtering, and building web apps in Python using Streamlit. The app is linked here (**[College Basketball App](https://cbb-basic-app.streamlit.app/)**), and more information about it is in the folder linked in the title above.

### App Description
In this app, you can
- Browse the full dataset of all Division I men's CBB teams and their stats for the season
- Filter down to a specific team with a dropdown
- Filter by one or multiple conferences with a multi-select
- Filter by number of wins with a range slider

### Motivation
The motivation for this project was to get hands-on experience building an interactive data app from scratch, while working with a dataset I am genuinely interested in as a basketball fan. It was a great exercise in presenting data interactively and interestingly that any basketball fan, no matter how little technical experience, can use and learn from. It was also a great way to get experience with Streamlit.

**Key skills:** Python, pandas, Streamlit, Interactive filtering and data display


## [Tidy Data Project](https://github.com/ldeend/Dee-Data-Science-Portfolio/tree/main/TidyData-Project)

### Project overview
In this project, I cleaned and transformed the mutant_moneyball.csv dataset using the tidy data principles. I took that dataset and used Python (pandas and matplotlib) to create pivot tables and create visualizations (with accompanying explanations). The original dataset was messy, with inconsistent formatting in cells, columns weren't stacked as they should be, and columns had multiple variables and labels in their names. All of this made the dataset difficult to analyze visually and programmatically. 

### Methodology
The data was reshaped using `melt()` and splitting the columns and column names with string operations such as `str.split()` or `str.replace()`. This new data was tidy, giving each variable its own unique column and each observation its own unique row. Finally, I used pivot tables and visualizations to show the most popular characters and the valuation across time and source.

### Motivation 
The motivation for this project is to build, refine, and showcase my data science abilities, specifically related to data cleaning and reshaping. This project complements my overall coding portfolio by highlighting various important skills, such as:
- understanding and applying foundational data science principles in tidy data
- cleaning and reshaping messy datasets 
- using pandas for dataset manipulation and aggregation 
- utilizing matplotlib to create professional and illustrative visualizations

**Key Skills:** Google Colab, data cleaning/reshaping, storytelling

## [Streamlit Supervised Machine Learning App](https://github.com/ldeend/Dee-Data-Science-Portfolio/blob/main/MLStreamlitApp)

### Project Overview

In this project, I built a live, interactive web application with Python and Streamlit where users can upload a dataset and train a supervised machine learning model on it (either Logistic Regression or Linear Regression, depending on the target variable used). The users can tune hyperparameters, change the train/test split, and simulate multiple times. There is no coding experience required, and all machine learning topics are explained in the app. Click on the title to go directly to the repo and learn more.

The app is linked here (**[Live ML App](https://lucasdee-mlapp.streamlit.app)**), and more information about it is in the section below.

## App Description
The app has these features:
- Accepts a CSV dataset uploaded by the user
- Supports Linear Regression (for continuous predictor variables) and Logistic Regression (for binary classification predictor variables)
- Includes hyperparameters that are all easily adjustable: regularization strength (α for Ridge, C for Logistic), penalty type (L1/L2), train/test split 
- Provides evaluation metrics and visualizations specific to each model (R², MSE/RMSE, MAE for regression and Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix, and ROC Curve for classification) to illustrate the models' capabilities
- Evaluation metrics and visualizations update automatically as the hyperparameters are changed, allowing the user to more easily experiment and see the impact of the hyperparameters
- Included guides for how to select which model and how to interpret the metrics

## Motivation
The motivation for this project was to reinforce what I've learned about supervised machine learning and get more experience with Streamlit. Not only did I have to code the app and make sure everything worked properly, but I also had to make it easily navigable to any user. This project forced me to understand the communication side of data science much better and required a deeper understanding of what I was actually doing when I was writing the code and why, all while keeping my code easy to follow. In that way, I think it adds to my overall portfolio by showcasing my **technical ability** with the app design and model building, as well as my **storytelling ability** with an engaging, easy-to-follow website.

**Key skills:** Python (scikit-learn, pandas, matplotlib, seaborn), Streamlit App design, Supervised Machine Learning (Logistic Regression and Linear Regression), Model Fitting and Evaluation


## [Streamlit Unsupervised Machine Learning App](https://github.com/ldeend/Dee-Data-Science-Portfolio/tree/main/MLUnsupervisedApp)

### Project Overview

In this project, I built a live, interactive web application with Python and Streamlit where users can upload a dataset and run an unsupervised machine learning model on it — either K-Means Clustering, Hierarchical Clustering, or PCA. Users can select feature variables, tune hyperparameters, and observe how their choices affect the model's output. There is no coding experience required, and all machine learning topics are explained in the app. Click on the title to go directly to the repo and learn more.

The app is linked here (**[Live Unsupervised ML App](https://lucasdee-unsupervisedmlapp.streamlit.app/)**), and more information about it is in the section below.

### App Description
The app has these features:
- Accepts a CSV dataset uploaded by the user
- Supports K-Means Clustering, Hierarchical Clustering, and PCA
- Includes hyperparameters that are all easily adjustable: number of clusters (k), initialization method, linkage method, distance metric, number of components, and more
- Provides evaluation metrics and visualizations specific to each model — Elbow Plot, Silhouette Analysis, and Cluster Scatter for clustering; Scree Plot, Cumulative Variance, and Component Loadings for PCA
- Evaluation metrics and visualizations update automatically as hyperparameters are changed, allowing the user to more easily experiment and see the impact of their choices
- Included guides for how to interpret every metric and visualization

### Motivation
This project was a natural continuation of the supervised ML app, pushing me to understand a fundamentally different class of machine learning — one with no target variable and no right answer to evaluate against. Building it forced me to think carefully about what "good" even means in an unsupervised setting, and how to communicate that clearly to a user with no ML background. Like the supervised app, it adds to my portfolio by showcasing both **technical ability** in building and deploying a real ML tool and **communication ability** in making complex concepts approachable.

**Key skills:** Python (scikit-learn, pandas, matplotlib, seaborn, scipy), Streamlit app design, Unsupervised Machine Learning (K-Means, Hierarchical Clustering, PCA), Model Evaluation and Visualization

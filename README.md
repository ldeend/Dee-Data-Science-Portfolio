# Data Science Portfolio

This repository is intended to showcase my data science projects and capabilities. I will have work related to data analysis, modeling, and machine learning.

## Planned Project Structure

This repository will be home to many different kinds of work. The primary concepts that will be present in these projects are `Exploratory Data Analysis`, `Data Cleaning and Visualization`, and `Statistical and Machine Learning Modeling`.

For each project, there will be a folder containing the dataset, methodology, results, insights, and steps for reproducibility in the README.


## [College Basketball Stats Streamlit App](https://github.com/ldeend/Dee-Data-Science-Portfolio/tree/main/basic_streamlit_app)

This is an interactive Streamlit app that allows users to look at and explore Division I Men's College Basketball teams and their stats via different filters. 
This project demonstrates skills in interactive data visualization, filtering, and building web apps in Python using Streamlit.  
Learn more about the app's features and how to launch it in the folder `basis_streamlit_app/`.


## [Tidy Data Project](https://github.com/ldeend/Dee-Data-Science-Portfolio/tree/main/TidyData-Project)

### Project overview
In this project, I cleaned and transformed the mutant_moneyball.csv dataset using the tidy data principles. I took that dataset and used Python (pandas and matplotlib) to create pivot tables and create visualizations. The original dataset was messy, with inconsistent formatting in cells, columns weren't stacked as they should be, and columns had multiple variables and labels in their names. All of this made the dataset difficult to analyze visually and programmatically. 

### Methodology
The data was reshaped using `melt()` and splitting the columns and column names with string operations such as `str.split()` or `str.replace()`. This new data was tidy, giving each variable its own unique column and each observation its own unique row. Finally, I used pivot tables and visualizations to show the most popular characters and the valuation across time and source.

### Motivation 
The motivation for this project is to build, refine, and showcase my data science abilities, specifically related to data cleaning and reshaping. This project complements my overall coding portfolio by highlighting various important skills, such as:
- understanding and applying foundational data science principles in tidy data
- cleaning and reshaping messy datasets 
- using pandas for dataset manipulation and aggregation 
- utilizing matplotlib to create professional and illustrative visualizations

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

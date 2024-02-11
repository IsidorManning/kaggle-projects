# Overview

This project aims to predict the survival of passengers aboard the Titanic using machine learning. This is probably one of the most famous datasets so I am very excited to showcase this project.

## The Dataset
The dataset used in this project is the famous Titanic dataset, which contains information about passengers such as their age, sex, class, ticket fare, and whether they survived or not. The dataset is split into a training set and a test set, with the training set containing information about a subset of passengers along with their survival status, and the test set containing similar information about another subset of passengers without their survival status.

# Project Structure
Data Loading and Initial Exploration: The project starts by importing necessary libraries and loading the Titanic dataset into pandas DataFrames. Initial exploration of the data, such as checking for missing values and understanding the distribution of variables, is performed.

## Data Preprocessing
This step involves preprocessing the data to handle missing values, outliers, and feature engineering. Missing values are imputed using appropriate strategies, outliers are detected and removed using Z-score, and new features are created based on existing ones.

## Exploratory Data Analysis (EDA)
The data is visually explored to understand the relationships between variables. Scatter plots, bar plots, and correlation heatmaps are used to visualize the data and identify patterns.

## Feature Engineering
Additional features are created or existing features are modified to improve the predictive power of the model. This includes creating new features like Fare/person and removing irrelevant features like Ticket_number.

# Model Building and Evaluation
Several machine learning models, including Random Forest, Extra Trees, and Decision Trees, are built and evaluated using cross-validation. Hyperparameter tuning is performed using GridSearchCV to optimize model performance.

### Model Deployment
The best-performing model is selected based on evaluation metrics, and predictions are generated on the test dataset. Submission files in CSV format are created for each model for submission to the Kaggle competition.

# Results

### Random Forest:
Train accuracy: 83.07% 
Validation accuracy: 80.06%

### Extra Trees:
Train accuracy: 83.99% 
Validation accuracy: 73.33%

### Decision Tree:
Train accuracy: 83.38% 
Validation accuracy: 78.78%

Conclusion
This was the first time I delve into feature engineering which is a crucial part in machine learning and data science. Initially, I had a really bad accuacy but I realized that feature engineering was key here to acheive an accuracy of above
80%.

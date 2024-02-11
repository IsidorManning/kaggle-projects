# Spaceship Titanic ML Project
This project aims to predict whether passengers on the Spaceship Titanic were transported successfully or not. The dataset contains information about various features such as age, spending on different amenities, home planet, destination, VIP status, etc.

## Problem Statement
The goal is to build machine learning models that can accurately predict whether a passenger was transported successfully or not based on the provided features.

## Evaluation Metric
Submissions are evaluated based on their classification accuracy, i.e., the percentage of predicted labels that are correct.

## Data Preprocessing
The following steps were taken to preprocess the data:

1. Handling missing values: Dropping rows with missing values in the "Name" and "Cabin" columns, dropping instances with more than 20% missing data, and imputing missing values in numerical features with mean and mode.
Feature engineering: Creating new features such as "CabinCount" and "FamilyCount" from existing features like "Cabin" and "Name".
2. Handling skewness: Applying cube root transformation to skewed numerical features.
3. Handling outliers: Removing outliers using Z-score method.
4. Categorical encoding: Using one-hot encoding for categorical features with low cardinality.
5. Scaling: Standard scaling of features.

## Exploratory Data Analysis (EDA)
Analyzed distribution and correlation of numerical features.
Plotted histograms, box plots, and correlation matrix to understand data distribution and relationships.
Visualized categorical features using strip plots to understand their distributions.

## Model Building and Training
Trained multiple machine learning models including Random Forest, CatBoost, XGBoost, KNN, and a Voting Classifier.

# Best Model Performance
### Random Forest:

Train accuracy: 73.18%
Validation accuracy: 72.91%

### CatBoost:

Train accuracy: 72.83%
Validation accuracy: 71.12%

### XGBoost:

Train accuracy: 73.06%
Validation accuracy: 70.38%

### KNN:

Train accuracy: 71.69%
Validation accuracy: 67.55%

### Voting Classifier (Ensemble of CatBoost, XGBoost, and Random Forest):

Train accuracy: 73.06%
Validation accuracy: 71.43%


# Conclusion & Future Improvements
The Random Forest model performed the best among the individual models, while the ensemble method showed promising results as well. Further tuning and experimentation could potentially improve model performance.

I will come back to this problem and try to imporve the accuracy at some points since I am not satisfied with the results. However, this project was great since it allowed me to enhance my skills when it comes to data preprocessing as well as EDA.

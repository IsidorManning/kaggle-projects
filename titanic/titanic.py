""" INITIAL NOTES

This is my full take on building a machine learning model which predicts the survival
of passengers of the titanic-data ship.

The goal: Certain groups of people had higher chances of surviving the titanic-data ship crash.
Our goal is to predict wether people will either survive or not survive based on certain
features. This is a binary classification problem.

"""

import pandas as pd  # For importing our data and visualizing our data in dataframes
import numpy as np  # For mathematical calculations and linear algebra

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

# Training & Evaluation
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Firstly we import and load all the necessary data from kaggle
train_url =  r"./titanic-data/train.csv"
test_url =  r"./titanic-data/train.csv"
train_df = pd.read_csv(train_url)  # Shape of (891, 12)
test_df = pd.read_csv(test_url)  # Shape of (418, 11)

# Drop the id column from both dataframes and store them in a separate variables
train_id = train_df["PassengerId"]
test_id = test_df["PassengerId"]
train_df = train_df.drop(["PassengerId"], axis=1)
test_df = test_df.drop(["PassengerId"], axis=1)

# Drop the target variable from the train dataframe and store in a separate variable
target = train_df["Survived"]
train_df = train_df.drop(["Survived"], axis=1)

# Fist we print the .info() and .describe() of both datasets to see if the datasets
# contain any missing values: We see that both the train and test datasets indeed contain
# missing values, but in slightly different columns. Let's put these columns in lists
nan_cols_train = ["Age", "Cabin", "Embarked"]
nan_cols_test = ["Age", "Fare", "Cabin", "Embarked"]

# y_train.isnull().any(): We can also see that the target varaible does not contain
# missing values which is good.

# Handling missing (nan) values.

# train_df.info() & test_df.info(): We can see that the feature "Cabin" contains a lot
# of missing values (about 77% for train_df and 80% for test_df) so we will remove this
# column completely.
train_df = train_df.drop(["Cabin"], axis=1)
test_df = test_df.drop(["Cabin"], axis=1)

# For the missing values in the feature "Age" we will simply replace it with the mean
train_df["Age"] = train_df["Age"].fillna(round(train_df["Age"].mean()))
test_df["Age"] = test_df["Age"].fillna(round(test_df["Age"].mean()))

# Since the amount of missing values in the feature "Embarked" is only 2, I will replace
# these 2 missing values with the most frequently reoccuring value for that feature.
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

# test_df["Fare"].iloc[152]: We can confirm that the instance in the test dataframe
# which has a missing value for the "Fare" variable has an index of 152. I also assume
# that fare is correlated with the class of the person (We will see if this is true
# later on when we do our EDA).
# test_df["Fare"].describe() & test_df.iloc[152]: We can see that the person at index
# 152 is in the lowest class (Pclass: 3). Therefore, I will replace the missing value for
# fare with the lower, 0.25, percentie of all values for fare in the test dataframe.
test_df.iloc[152] = test_df.iloc[152].fillna(7.8958)

# train_df.isnull().any().sum() & test_df.isnull().any().sum(): Our datasets does not
# contain any missing values anymore!

# EDA - Unvariate Analysis

# First, lets examine our previus assumption that the variable "Fare" is positively
# correlated with the feature "Pclass" and see if we can confirm the assumption.

# plt.scatter(train_df["Fare"], train_df["Pclass"]): We can see that the assumption is
# partially right but not for all instances. However, since the missing value only occurs
# in one single instace, I will stick with the imputation tecnique I used.

# Let's check the skewness of each numerical feature!
numericals = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
skewness = train_df[numericals].skew()

# plt.bar(skewness.index, skewness.values): We can see that "Fare" is very right skewed.
# I will perform a log-1p transformation on this column both on the train and test dataset.
# I will not perform any transformation on any other feature with a skewness since they
# are not continoues features, unlike Fare.
train_df["Fare"] = np.log1p(train_df["Fare"])

# Let's remove some outliers from the training dataset with the help of Z score.
train_z_scores = zscore(train_df[numericals])
train_outliers = np.where(np.abs(train_z_scores) > 3)[0]
train_df = train_df.drop(train_outliers, axis=0)
target = target.drop(train_outliers, axis=0)

# Since the categorical columns "Name" and "Ticket" contain many unique values,
# we will create some simple functions which will preprocess these columns and create
# some new ones.
suffixes = ["Rev", "Capt", "Mr", "Miss", "Mrs", "Mme", "Master", "van", "Dr", "Col",
            "Mlle", "Major", "Ms", "Lady", "Sir", "Don", "Countess", "Jonkheer",
            "Dona", "Van"]


def normalize_name(x):
    return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])


def extract_name_suffix(x):
    for split in x.split(" "):
        if split in suffixes:
            if split in ['Mlle', 'Ms', "Miss", "Dona"]:
                return 'Miss'
            elif split in ['Lady', "Mrs", 'Mme', 'Countess']:
                return 'Mrs'
            elif split in ['Sir', 'Don', "Mr", "van", "Van"]:
                return 'Mr'
            elif split in ["Master", "Jonkheer"]:
                return 'Master'
            else:  # ['Dr', 'Rev', 'Major', 'Col', 'Capt']:
                return 'Other'


def remove_suffixes_in_name_columns(x):
    splitted = [v.strip(",()[].\"'") for v in x.split(" ")]
    for i in range(len(splitted)):
        if splitted[i] in suffixes:
            splitted.pop(i)
            break
    return " ".join(splitted)


def ticket_number(x):
    splitted = x.split(" ")[-1]
    if splitted == "LINE":
        return 0
    else:
        return int(splitted)


def ticket_item(x):
    items = x.split(" ")
    if len(items) == 1:
        return "NONE"
    return "_".join(items[0:-1])


class preprocess_name_and_ticket:
    def __init__(self, x):
        self.df = x

    def preprocess_name(self):
        self.df["Name"] = self.df["Name"].apply(normalize_name)
        self.df["Name_suffix"] = self.df["Name"].apply(extract_name_suffix)
        self.df["Name"] = self.df["Name"].apply(remove_suffixes_in_name_columns)

    def preprocess_ticket(self):
        self.df["Ticket_number"] = self.df["Ticket"].apply(ticket_number)
        self.df["Ticket_item"] = self.df["Ticket"].apply(ticket_item)


preprocess_name_and_ticket(train_df).preprocess_name()
preprocess_name_and_ticket(train_df).preprocess_ticket()
train_df = train_df.drop(["Name", "Ticket"], axis=1)

preprocess_name_and_ticket(test_df).preprocess_name()
preprocess_name_and_ticket(test_df).preprocess_ticket()
test_df = test_df.drop(["Name", "Ticket"], axis=1)

# Visualizing categorical features and their distribution with bar plots

# Now would be a good time to also check the frequency distribution on all categorical
# features.

categoricals = ["Name_suffix", "Sex", "Embarked", "Ticket_item"]
numericals.append("Ticket_number")


def bar_multiple(shape):
    i = -1
    col_dim, row_dim = shape[0], shape[1]
    fig, ax = plt.subplots(col_dim, row_dim, figsize=(17, 8))
    for row_plot in range(0, col_dim):
        for col_plot in range(0, row_dim):
            i += 1
            counts = train_df[categoricals[i]].value_counts()
            labels = counts.index
            ax[row_plot, col_plot].bar(labels, counts)
            ax[row_plot, col_plot].title.set_text(categoricals[i])
    plt.show()


# EDA - Multivariate analysis

# Let's plot a correlation heatmap to see which variables have a relationship between them.

corr = pd.concat([train_df[numericals], target], axis=1).corr()
# sns.heatmap(data=corr): We can see that the column "Fare" have pretty strong positive
# correlations with the columns "Parch", "SibSp" and our target "Survived" which makes
# sense. The more money you have and have spent on your trip on titanic-data,
# the more likely you are to have a larger family with more children and siblings and the
# more money you have, the more likely you are to have survived. We can now conclude
# that these columns and especially "Fare" are very important to us. We also see that
# Ticket_number is correlated with Pclass. Since Fare is so important, we will now create
# a new column which will be fare/person
train_df["Fare/person"] = train_df["Fare"] // (train_df["SibSp"] + train_df["Parch"] + 1)
test_df["Fare/person"] = test_df["Fare"] // (test_df["SibSp"] + test_df["Parch"] + 1)
numericals.append("Fare/person")

# plt.scatter(x=train_df["Ticket_number"], y=target): I am a bit suspicious of the
# "Ticket_number" column. Let's plot it against survival rate to see if it makes
# any sense to keep this feature. After the plotting, we can clearly see that ticket
# number did not play any sort of roll when it comes to deciding if one survives or not.
# Therefore, we will drop this column.
train_df = train_df.drop(["Ticket_number"], axis=1)
test_df = test_df.drop(["Ticket_number"], axis=1)
numericals.pop(numericals.index("Ticket_number"))

# Encoding categorical features
train_df['Ticket_Group'] = train_df.groupby('Ticket_item')['Ticket_item'].transform(
    'count')
train_df.drop(['Ticket_item'], axis=1, inplace=True)
test_df['Ticket_Group'] = test_df.groupby('Ticket_item')['Ticket_item'].transform('count')
test_df.drop(['Ticket_item'], axis=1, inplace=True)
categoricals.pop(categoricals.index("Ticket_item"))

# Encode all variables with the help of column transformer
scaler = StandardScaler(with_mean=True)
encoder = OneHotEncoder(sparse_output=False)
categorical_pipe = make_pipeline(encoder, scaler)

# Make column transformer
ct = make_column_transformer(
    (scaler, numericals),
    (categorical_pipe, categoricals),
)

# Transform data
train_preprocessed = ct.fit_transform(train_df)
test_preprocessed = ct.transform(test_df)

# Training

# Split up the train dataset into train and validation subsets
X_train, X_val, y_train, y_val = train_test_split(train_preprocessed, target,
                                                  test_size=0.2,
                                                  random_state=42)


# Create models
def custom_model_maker(estimator, name):
    estimator.fit(X_train, y_train)
    score_train = cross_val_score(estimator, X_train, y_train, cv=5, n_jobs=-1,
                                  scoring="accuracy").mean()
    score_val = cross_val_score(estimator, X_val, y_val, cv=5, n_jobs=-1,
                                scoring="accuracy").mean()

    print(f"{name} score on train: {score_train}")
    print(f"{name} score on val: {score_val}")
    print("")


verbose = 0
cv = 5

# Random Forests
params_rf = {"n_estimators": [100, 120, 90, 120],
             "min_samples_split": [20, 40, 50],
             "max_depth": [10, 20, 40],
             "criterion": ("gini", "log_loss", "entropy")}
rf = RandomForestClassifier()
gd_rf = GridSearchCV(rf, param_grid=params_rf, cv=cv, verbose=verbose)
gd_rf.fit(X_train, y_train)
print(f"Best parameters rf: {gd_rf.best_params_}")
custom_model_maker(gd_rf.best_estimator_, "rf")

# Extra trees
params_extra = {"n_estimators": [100, 120, 150, 140],
                "min_samples_split": [5, 10, 20, 40, 60],
                "max_depth": [4, 10, 20, 40],
                "criterion": ("gini", "log_loss", "entropy")}
extra_rf = ExtraTreesClassifier()
gd_extra = GridSearchCV(extra_rf, param_grid=params_extra, cv=cv, verbose=verbose)
gd_extra.fit(X_train, y_train)
print(f"Best parameters extra : {gd_extra.best_params_}")
custom_model_maker(gd_extra.best_estimator_, "extra trees")

# Decision tree
params_tree = {"min_samples_leaf": [1, 5, 10, 20, 50],
               "min_samples_split": [60, 70, 80, 100],
               "max_depth": [4, 10, 20, 40],
               "criterion": ("gini", "log_loss", "entropy")}
tree_clf = DecisionTreeClassifier()
gd_tree = GridSearchCV(tree_clf, param_grid=params_tree, cv=cv, verbose=verbose)
gd_tree.fit(X_train, y_train)
print(f"Best parameters dt : {gd_tree.best_params_}")
custom_model_maker(gd_tree.best_estimator_, "decision tree")

# Submissions

best_estimator_1 = gd_rf.best_estimator_
y_pred_1 = best_estimator_1.predict(test_preprocessed)

# Create submission file 1
submission_1 = pd.DataFrame({
    "PassengerId": test_id,
    "Survived": y_pred_1
})
submission_1.to_csv("titanicsubmissionrf.csv", index=False)

best_estimator_2 = gd_extra.best_estimator_
y_pred_2 = best_estimator_2.predict(test_preprocessed)

# Create submission file 2
submission_2 = pd.DataFrame({
    "PassengerId": test_id,
    "Survived": y_pred_2
})

submission_2.to_csv("titanicsubmissionextra.csv", index=False)

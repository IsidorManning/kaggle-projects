"""
Notes:

Submissions are evaluated based on their classification accuracy,
the percentage of predicted labels that are correct.

"""

# For calculations & storing our data
import pandas as pd
import numpy as np
import math
from scipy.stats import skew

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

# Training & Evaluation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Misc
import warnings

warnings.filterwarnings("ignore")

# Load data into train & test dataframes:

train_url = r"./spaceship/train.csv"
train = pd.read_csv(train_url)

test_url = r"./spaceship/test.csv"
test = pd.read_csv(test_url)


# Create two functions which we will use a lot:

def get_numericals(df):
    return df.select_dtypes(include=['int16', 'int32', 'int64',
                                     'float16', 'float32', 'float64']).columns


def get_categoricals(df):
    return df.select_dtypes(include=['object']).columns


# Handle null values:

# All columns contain about 200 missing values. We are going to drop rows containing null
# values in columns Name & Cabin since imputating these features based on some
# statistical measurment would produce unreliable predictions.

train = train.dropna(subset=["Name", "Cabin"])
test = test.dropna(subset=["Name", "Cabin"])

# Before we handle all null values we first want to drop instances containg, in our case,
# more than 20% missing data. Performing imputation on these types of isntances would,
# again, produce unreliable predictions.

train = train.dropna(thresh=train.shape[1] * 0.80, axis=0).reset_index(drop=True)
test = test.dropna(thresh=test.shape[1] * 0.80, axis=0).reset_index(drop=True)


def simple_null_handler(df):
    """
    Handle null values contained within all numerical features except for "VIP" with
    either the statistical measurements mean and mode, depending on the context of the
    feature
    """
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["RoomService"] = df["RoomService"].fillna(df["RoomService"].mean())
    df["FoodCourt"] = df["FoodCourt"].fillna(df["FoodCourt"].mean())
    df["ShoppingMall"] = df["ShoppingMall"].fillna(df["ShoppingMall"].mean())
    df["Spa"] = df["Spa"].fillna(df["Spa"].mean())
    df["VRDeck"] = df["VRDeck"].fillna(df["VRDeck"].mean())
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode().iloc[0])
    df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].mode().iloc[0])
    df["Destination"] = df["Destination"].fillna(df["Destination"].mode().iloc[0])


simple_null_handler(train)
simple_null_handler(test)


# Feature engineering:

def create_cabin_count_from_cabin(df):
    """
    Creates a new column "CabinCount" from the original column "Cabin" which contains
    a number for every instance in the dataset indicating how many people live in the
    same cabin as that observed instance does.
    """
    unique_cabins_count = df["Cabin"].value_counts()
    df["CabinCount"] = np.nan
    cabin_count_data = []
    for i in range(df.shape[0]):
        current_cabin = df["Cabin"].iloc[i]
        cabin_count_data.append(unique_cabins_count.loc[current_cabin])

    df["CabinCount"] = cabin_count_data


create_cabin_count_from_cabin(train)
create_cabin_count_from_cabin(test)

# Now that we have extracted the valuable data from the "Cabin" feature and created a new
# feature, we will simply delete the variable.

train = train.drop(["Cabin"], axis=1)
test = test.drop(["Cabin"], axis=1)


# By examening, one can find that all instances within the "Name" feature only has
# a first and a last name. So while "Name" by itself might not be useful to us, we can
# definetly extract some useful information.

def create_family_count_from_name(df):
    """
    Creates a new column "FamilyCount" from the original column "Name" which contains
    a number for every instance in the dataset indicating how many people have the same
    last name as the instance itself has (i.e., how many family members the instance is
    traveling with).
    """
    memo = {}

    for i in range(df.shape[0]):
        last_name = df["Name"].iloc[i].split()[1]
        if last_name not in memo:
            memo[last_name] = 1
        else:
            memo[last_name] += 1

    df["FamilyCount"] = df["Name"].apply(lambda name: memo[name.split()[1]])


create_family_count_from_name(train)
create_family_count_from_name(test)

# Now that we have extracted the valuable data from the "Name" feature and created a new
# feature, we will simply delete the variable.

train = train.drop(["Name"], axis=1)
test = test.drop(["Name"], axis=1)

# Now that we have removed certain columns and rows we can extract the dataframe ids &
# the target variable itself.

target = train["Transported"]
train_id = train["PassengerId"]
train = train.drop(["Transported", "PassengerId"], axis=1)
test_id = test["PassengerId"]
test = test.drop(["PassengerId"], axis=1)


# EDA - Continous variables:


def plot_numerical_features(df, dim, num_cols=4):
    """
    Plots all numerical features of the inputted dataset (df) where each plot is plotting
    the index value of every instance against its corresponding value for the current
    feature.
    """
    features = get_numericals(df)
    total_plots = len(features)
    num_rows = (total_plots + num_cols - 1) // num_cols
    plt.figure(figsize=dim)
    colors = plt.cm.rainbow(np.linspace(0, 1, total_plots))

    for i, (feature, color) in enumerate(zip(features, colors), start=1):
        plt.subplot(num_rows, num_cols, i)
        plt.plot(df[feature], marker='o', linestyle='None', color=color)
        plt.title(feature)
        plt.xlabel('Index')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.show()


# With the use of our "plot_numerical_features" function we can see that there are not
# many outliers and recording errors within this specific dataset. There are only a few
# smaller outliers in "RoomService", "FoodCourt", "ShoppingMall", "Spa" and "VRDeck".


def plot_distribution(df, dim, bins=20):
    """
    Plot multiple histograms in separate plots for each feature.
    """
    df.hist(bins=bins, figsize=dim, layout=(-1, 5), edgecolor="black")
    plt.tight_layout()
    plt.show()


# With the use of our "plot_distribution" function we can see that most of our numerical
# features are right-skewed where the features "RoomService", "FoodCourt", "ShoppingMall",
# "Spa", "VRDeck" and "CabinCount" are heavly right-skewed.


def boxplot_numerical_variables(df):
    """
    Boxplot all the numerical features within the inputted dataset (df)
    """
    features = get_numericals(df)
    num_features = len(features)
    num_cols = 3
    num_rows = math.ceil(num_features / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    for idx, feature in enumerate(features):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        ax = axes[row_idx, col_idx]

        sns.boxplot(data=df, x=feature, ax=ax)
        ax.set_title(f'Boxplot of {feature}')

    plt.tight_layout()
    plt.show()


def plot_covarience_matrix(df):
    """
    Plots a correlation matrix of the continous features from the inputted dataset (df).
    """
    features = get_numericals(df)
    correlation_df = df[features].corr()
    mask = np.triu(np.ones_like(correlation_df, dtype=bool))
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_df, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


# With the use of our "plot_covarience_matrix" function we can see that most of our
# continous features does not have any extremely strong correlation with eachother.
# The correlation ranges between -0.2 and 0.3. "CabinCount" has a positive correlation
# with "FamilyCount" (which makes sense), "Age" has a somewhat negative correlation with
# both "FamilyCount" and "CabinCount", "VRDeck" has a somewhat positive correlation with
# "FoodCourt" and "Spa" has a somewhat positive correlation with "FoodCourt" also. Other
# than these pinpoints, the correlation matrix does not seem to provide any useful
# information for us.

# EDA - Ordinal variables:


def stripplot_ordinal_features(df, n_cols=4):
    """
    Stripplot every single categorical feature within the inputted dataset (df) to
    a specified continous variable to spread out the categorical data.
    """
    df_ordinal = get_categoricals(df)
    n_elements = len(df_ordinal)
    n_rows = np.ceil(n_elements / n_cols).astype("int")
    y_value = df["Age"]  # Specify y_value to spread data (ideally a continuous feature)

    fig, axes = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(15, n_rows * 2.5))

    for col, ax in zip(df_ordinal.columns, axes.ravel()):
        sns.stripplot(data=df, x=col, y=y_value, ax=ax,
                      palette="tab10", size=1, alpha=0.5)
    plt.tight_layout()
    plt.show()


# With the use of our "stripplot_ordinal_features" function we can see that all of our
# categorical features have a pretty low cardionality. "HomePlanet" has 3 values,
# Destination also has 3 values and "VIP" has 2 values (Notice that there are still some
# null values in the "VIP" feature which we have not dealt with yet. We will get to
# handling these null values later when we start to encode our categorical variables).

# Some more feature engineering:

# After some time investigating the features of the dataset, we can see that "Spa",
# "FoodCourt", "RoomService", "ShoppingMall", "VRDeck" almsot have the exact same
# distribution and correlation. What I will is to create a new column which conacatenates
# all of these column's value into one column.

def create_mean_spent_variable(df):
    """
    Create a new column called "MeanSpent" which shows the avarege amount of money
    that the instances have spent on the trip on FoodCourt, RoomsSrvice, the VRDeck,
    on Spa and in the ShoppinMall.
    """
    spending_columns = ["VRDeck", "FoodCourt", "Spa", "ShoppingMall", "RoomService"]
    df["MeanSpent"] = df[spending_columns].mean(axis=1)


create_mean_spent_variable(train)
create_mean_spent_variable(test)

train = train.drop(["VRDeck", "FoodCourt", "Spa", "ShoppingMall",
                    "RoomService"], axis=1)
test = test.drop(["VRDeck", "FoodCourt", "Spa", "ShoppingMall",
                  "RoomService"], axis=1)


# Now we'll handle the null values within the "VIP" feature based on our new "MeanSpent".
# I assume that there is a positive relationship between the features "VIP" and
# # "RoomService", "FoodCourt", "ShoppingMall", "Spa" "VRDeck" (The more money you spend
# # on your trip, the more money you have and the more likely you are to be a VIP quest).


def null_handler_for_VIP(df):
    # Calculate the third quantile of the "MeanSpent" column
    third_quantile = df['MeanSpent'].quantile(0.75)

    # Iterate through the rows with missing "VIP" values
    for index, row in df[df['VIP'].isna()].iterrows():
        mean_spent_value = row['MeanSpent']

        # Impute "VIP" based on "MeanSpent" value
        if mean_spent_value >= third_quantile:
            df.at[index, 'VIP'] = True
        else:
            df.at[index, 'VIP'] = False

    return df


train = null_handler_for_VIP(train)
test = null_handler_for_VIP(test)


# Handling skewness:

# In the EDA we saw that almost all of our numerical features are skewed to the left.
# We will now investigate this further and fix this issue.


def get_skew_for_all_numerical_features(df):
    features = get_numericals(df)
    result = {}
    for feature in features:
        result[f"{feature}_skew"] = skew(df[feature])
    return result


skew_of_train = get_skew_for_all_numerical_features(train)
skew_of_test = get_skew_for_all_numerical_features(train)

# As we saw, all of our numerical features are heavely right-skewed, and we can
# confirm this by printing the skewness value for all these features. Let's fix this by
# taking the cube root of all skewed features which, in our case, are all of our
# numerical features.

skewed_features = get_numericals(train)
train[skewed_features] = np.cbrt(train[skewed_features])
test[skewed_features] = np.cbrt(test[skewed_features])

# Handling outliers:

# We saw that there were some outliers in our dataset before in the EDA phase but not too
# many at all. So let's now remove some of these outliers from the training dataset with
# the help of Z-score. Let's also keep the Z-score threshold pretty high so that we don't
# accedantly remove instances that actually aren't outliers.

z_score_threshold = 4
numericals = get_numericals(train)
train_z_scores = zscore(train[numericals])
train_outliers = np.where(np.abs(train_z_scores) > z_score_threshold)[0]
train = train.drop(train_outliers, axis=0)
target = target.drop(train_outliers, axis=0)

# Categorical encoding:

# Becase of the fact that our categorical features have a low cardionality, we will
# encode them using one hot encoding

ohe = OneHotEncoder(sparse_output=False)

# Encoding the train dataset with the one hot encoder
cats = get_categoricals(train)
nums = get_numericals(train)

encoded_data_train = ohe.fit_transform(train[cats])
encoded_df_train = pd.DataFrame(encoded_data_train,
                                columns=ohe.get_feature_names_out(cats))
train_encoded = pd.concat([train[nums].reset_index(drop=True),
                           encoded_df_train.reset_index(drop=True)], axis=1)

# Encoding the test dataset using the fitted one hot encoder used on the train dataset

encoded_data_test = ohe.transform(test[cats])
encoded_df_test = pd.DataFrame(encoded_data_test,
                               columns=ohe.get_feature_names_out(cats))
test_encoded = pd.concat([test[nums].reset_index(drop=True),
                          encoded_df_test.reset_index(drop=True)], axis=1)

target = LabelEncoder().fit_transform(target)

# Scaling:

scaler = StandardScaler(with_mean=True)

# Scaling the train dataset with the standard scaler

train_preprocessed = pd.DataFrame(scaler.fit_transform(train_encoded),
                                  columns=scaler.get_feature_names_out())

# Scaling the test dataset using the fitted scaler used on the train dataset

test_preprocessed = pd.DataFrame(scaler.transform(test_encoded),
                                 columns=scaler.get_feature_names_out())

# Training:

# Split up the train dataset into train and validation subsets

X_train, X_val, y_train, y_val = train_test_split(train_preprocessed, target,
                                                  test_size=0.2,
                                                  shuffle=True)


# Create training models


def custom_model_maker(estimator, name):
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

params_rf = {
    "n_estimators": [100, 120, 90, 160],
    "min_samples_split": [20, 40, 50, 80],
    "max_depth": [10, 20, 40],
}

rf = RandomForestClassifier(criterion="log_loss")
gd_rf = GridSearchCV(rf, param_grid=params_rf, cv=cv, verbose=verbose)
gd_rf.fit(X_train, y_train)
print(f"Best parameters rf: {gd_rf.best_params_}")
custom_model_maker(gd_rf.best_estimator_, "Random Forest")

# Cat boost

params_cat_boost = {
    "iterations": 604,
    "learning_rate": 0.008574858135413305,
    "depth": 6,
    "l2_leaf_reg": 0.024986714637090388,
    "bootstrap_type": "Bayesian",
    "random_strength": 3.700277878995026e-05,
    "bagging_temperature": 0.027325609307599474,
    "od_type": "Iter",
    "od_wait": 39,
}
cat_boost = CatBoostClassifier(**params_cat_boost, verbose=0)
cat_boost.fit(X_train, y_train)
custom_model_maker(cat_boost, "Cat boost")

# XGBoost

params_xgb = {
    "n_estimators": [100, 120, 90],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [10, 6, 15],
}

xgb = XGBClassifier()
gd_xgb = GridSearchCV(xgb, param_grid=params_xgb, cv=cv, verbose=0)
gd_xgb.fit(X_train, y_train)
print(f"Best parameters rf: {gd_xgb.best_params_}")
custom_model_maker(gd_xgb.best_estimator_, "XGB")

# KNN

params_knn = {
    "n_neighbors": [2, 5, 10, 20, 50, 80],
    "leaf_size": [15, 30, 60],
}

knn = KNeighborsClassifier()
gd_knn = GridSearchCV(knn, param_grid=params_knn, cv=cv, verbose=verbose)
gd_knn.fit(X_train, y_train)
print(f"Best parameters knn : {gd_knn.best_params_}")
custom_model_maker(gd_knn.best_estimator_, "KNN")

# Voting

voting_classifier = VotingClassifier(
    estimators=[
        ('catboost', cat_boost),
        ('XGB', gd_xgb.best_estimator_),
        ('RF', gd_rf.best_estimator_)
    ],
    voting='soft'
)
voting_classifier.fit(X_train, y_train)
custom_model_maker(voting_classifier, "Voting clf (Catboost, XGB & RF)")

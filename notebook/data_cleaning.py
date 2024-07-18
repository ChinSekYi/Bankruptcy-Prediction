import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.metrics import r2_score

"""
functions starting with df_ can generate a processed dataframe directly
"""


# TODO: add exception
# function to convert target column to binary values 0 and 1
class AsDiscrete(BaseEstimator, TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        ncol = len(df.columns) - 1
        feature_space = df.iloc[:, 0:ncol]
        target_column = df.iloc[:, ncol]

        n = len(target_column)
        new_col = [0] * n
        for i in range(n):
            if target_column[i] == "b'0'":
                new_col[i] = 0
            else:
                new_col[i] = 1

        pd_col = pd.DataFrame(new_col, columns=["class"])
        new_df = pd.concat([feature_space, pd_col], axis=1)

        new_df["class"] = new_df["class"].astype("category")

        return new_df


# Define a function for mapping
def map_class_labels(df):
    mapping = {0: "not-bankrupt", 1: "bankrupt"}
    df["class"] = df["class"].map(mapping)
    return df


# function to separate features and target
def get_Xy(df):
    X = df.iloc[:, 0 : len(df.columns) - 1]
    y = df.iloc[:, -1]
    return X, y


# function to handle missing values
def med_impute(X, y):
    # Remove columns with more than 40% null values
    thd1 = X.shape[0] * 0.4
    cols = X.columns[X.isnull().sum() < thd1]
    X = X[cols]

    # Remove rows with more than 50% null values
    thd2 = X.shape[1] * 0.5
    y = y[X.isnull().sum(axis=1) <= thd2]
    X = X[X.isnull().sum(axis=1) <= thd2]

    # Median imputation for remaining null values
    X = X.fillna(X.median())
    return X, y


# function to normalise numerical columns to remove effect of inconsistent scales
def normalise(df):
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return X_scaled


def check_skewness(df):
    X, y = get_Xy(df)
    skewness = X.skew()
    skewed_features = skewness[abs(skewness) > 0.5]
    num_skewed_features = len(skewed_features)
    return num_skewed_features


def count_and_percentage_outliers(df):
    X, y = get_Xy(df)
    outlier_counts = {}
    outlier_percentages = {}

    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]):
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)]

            outlier_counts[column] = len(outliers)
            percentage_outliers = (len(outliers) / len(X)) * 100
            outlier_percentages[column] = (
                percentage_outliers  # Format percentage with two decimal places and add '%' sign
            )

    outlier_counts_df = pd.DataFrame(
        list(outlier_counts.items()), columns=["Column", "Number of Outliers"]
    )
    outlier_percentages_df = pd.DataFrame(
        list(outlier_percentages.items()), columns=["Column", "Percentage of Outliers"]
    )

    result_df = pd.concat(
        [outlier_counts_df, outlier_percentages_df["Percentage of Outliers"]], axis=1
    )

    return result_df


# functions to solve skewed data
def log_transform(df):
    X, y = get_Xy(df)
    # Apply log transformation to numeric columns only
    X_transformed = X.select_dtypes(include=[np.number]).apply(
        lambda x: np.log(x + 1)
    )  # Adding 1 to avoid log(0)
    df_transformed = pd.concat([X_transformed, y], axis=1)
    return df_transformed


def sqrt_transform(df):
    X, y = get_Xy(df)
    X_transformed = X.select_dtypes(include=[np.number]).apply(lambda x: np.sqrt(x))
    df_transformed = pd.concat([X_transformed, y], axis=1)
    return df_transformed


def cube_root_transform(df):
    X, y = get_Xy(df)
    X_transformed = X.select_dtypes(include=[np.number]).apply(lambda x: np.cbrt(x))
    df_transformed = pd.concat([X_transformed, y], axis=1)
    return df_transformed


def boxcox_transform(df):
    X, y = get_Xy(df)
    X_transformed = X.select_dtypes(include=[np.number]).apply(
        lambda x: boxcox(x + 1)[0] if np.all(x > 0) else x
    )  # Box-Cox requires positive values
    df_transformed = pd.concat([X_transformed, y], axis=1)
    return df_transformed


def remove_outliers_mad(df, column, threshold=3.5):
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Apply MAD method
def apply_mad_removal(df):
    X, y = get_Xy(df)
    for column in X.columns:
        X_cleaned = remove_outliers_mad(X, column)

    X_cleaned.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    df_concat = pd.concat([X_cleaned, y], axis=1)

    return df_concat


### overall
def df_cleaning(df):
    X, y = get_Xy(df)
    X_imputed, y = med_impute(X, y)
    X_scaled_df = normalise(X_imputed)
    y_df = pd.DataFrame(y)

    X_scaled_df.reset_index(drop=True, inplace=True)
    y_df.reset_index(drop=True, inplace=True)

    df_concat = pd.concat([X_scaled_df, y_df], axis=1)
    df_concat["class"] = df_concat["class"].astype(int)
    result_df = map_class_labels(df_concat)
    return result_df


def df_preprocess_after_EDA(df):
    column_names = pd.read_csv("column_names.txt", header=None)
    df.columns = column_names
    df_cleaned = df_cleaning(df)
    df_cube_root_transformed = cube_root_transform(df_cleaned)
    df = df_cube_root_transformed
    return df


### Used in FEATURE_SELECTION


# preliminary cleaning
def df_null_removal(df):
    X, y = get_Xy(df)
    X_imputed, y = med_impute(X, y)
    X_scaled_df = normalise(X_imputed)
    return X_scaled_df, y


# function for feature selection
def drop_high_corr(X, threshold=0.7):
    correlation_matrix = X.corr()
    high_cor = []
    dropped_features = []

    # Iterate through the correlation matrix to find highly correlated pairs
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                if correlation_matrix.columns[j] != correlation_matrix.columns[i]:
                    high_cor.append(
                        [
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j],
                        ]
                    )

    # Iterate through the list of highly correlated pairs
    for pair in high_cor:
        feature1, feature2, correlation = pair

        # Check if either of the features in the pair has already been dropped
        if feature1 not in dropped_features and feature2 not in dropped_features:
            # Check if the feature exists in the DataFrame before attempting to drop it
            if feature2 in X.columns:
                # Drop one of the correlated features from the dataset
                # Here, we arbitrarily choose to drop the second feature in the pair
                X.drop(feature2, axis=1, inplace=True)
                dropped_features.append(feature2)
            else:
                print("Feature '" + feature2 + "' not found in the DataFrame.")

    X.reset_index(drop=True, inplace=True)
    return X, dropped_features


def drop_corr_columns_from_test(X_test, dropped_features):
    X_test_dropped = X_test.drop(columns=dropped_features, errors="ignore")
    return X_test_dropped


# secondary cleaning
def df_null_corr_process(df):
    X, y = df_null_removal(df)
    return drop_high_corr(X), y


# function to obtain train and test sets
def get_train_test(df):
    X, y = df_null_corr_process(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3244
    )

    return X_train, X_test, y_train, y_test


# function to obtain train and test sets with sythesised instances of the minority class
def pre_process(df):
    X, y = df_null_corr_process(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1000
    )
    smote = SMOTE(random_state=0)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    return X_smote, X_test, y_smote, y_test


def plot_ANOVA_test_graph(train_acc_dict, test_acc_dict):
    # Extract keys and values from train_acc_dict and test_acc_dict
    train_k_values, train_accuracy_values = zip(*train_acc_dict.items())
    test_k_values, test_accuracy_values = zip(*test_acc_dict.items())

    plt.figure(figsize=(6, 4))
    # Plot train accuracy
    plt.plot(
        train_k_values, train_accuracy_values, label="Train Accuracy", color="blue"
    )
    # Plot test accuracy
    plt.plot(test_k_values, test_accuracy_values, label="Test Accuracy", color="green")

    # Find k values corresponding to maximum accuracies
    best_train_k = max(train_acc_dict, key=train_acc_dict.get)
    best_test_k = max(test_acc_dict, key=test_acc_dict.get)
    best_train_accuracy = train_acc_dict[best_train_k]
    best_test_accuracy = test_acc_dict[best_test_k]

    # Annotate the point corresponding to the peak train accuracy
    plt.annotate(
        f"Max Train Accuracy\nk={best_train_k}, Acc={best_train_accuracy:.2f}",
        xy=(best_train_k, best_train_accuracy),
        xytext=(-30, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    # Annotate the point corresponding to the peak test accuracy
    plt.annotate(
        f"Max Test Accuracy\nk={best_test_k}, Acc={best_test_accuracy:.2f}",
        xy=(best_test_k, best_test_accuracy),
        xytext=(30, -30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    # Label axes and add title
    plt.xlabel("Number of Features (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Features from ANOVA test")

    plt.legend()
    plt.show()


## Note: *args follow the convention X_train, X_test, y_train, y_test
def get_df_with_top_k_features(k_features, *args):  # after pre_process(df)
    X_train = args[0]
    X_test = args[1]
    y_train = args[2]
    y_test = args[3]

    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=k_features)

    # apply feature selection
    fs.fit_transform(X_train, y_train)

    # Take the features with the highest F-scores
    fs_scores_array = np.array(fs.scores_)

    # Get the indices that would sort the array in descending order
    sorted_indices_desc = np.argsort(fs_scores_array)[::-1]

    # Take the top k indices
    top_indices = sorted_indices_desc[:k_features]

    selected_columns_X_train = X_train.iloc[:, top_indices]
    selected_columns_X_test = X_test.iloc[:, top_indices]

    return selected_columns_X_train, selected_columns_X_test, y_train, y_test


def find_best_k_features_from_ANOVA(model, *args):
    X_train = args[0]
    original_n_features = len(X_train.columns)

    # find the optimum number of features that gives the best test accuracy
    train_acc_dict = {}  # 0 is a dummy accuracy for k=0 features
    test_acc_dict = {}
    train_test_dataset = {}

    for k in range(1, original_n_features + 1):
        train_test_dataset_after_ANOVA = get_df_with_top_k_features(k, *args)
        train_accuracy, test_accuracy = model(*train_test_dataset_after_ANOVA)
        train_test_dataset[k] = train_test_dataset_after_ANOVA
        train_acc_dict[k] = train_accuracy
        test_acc_dict[k] = test_accuracy

    # Find k that gives the highest accuracy
    best_train_k = max(train_acc_dict, key=train_acc_dict.get)
    best_test_k = max(test_acc_dict, key=test_acc_dict.get)

    print(f"\033[96mBest k for train_accuracy:\033[00m {best_train_k}")
    print(f"\033[96mBest k for test_accuracy:\033[00m {best_test_k}")

    plot_ANOVA_test_graph(train_acc_dict, test_acc_dict)

    return train_test_dataset[best_test_k]


def evaluate_model(model, params, *args):
    """
    Evaluate multiple models using GridSearchCV and return their R-squared scores.

    Args:
        x_train (array-like): Training input samples.
        y_train (array-like): Target values for training.
        x_test (array-like): Test input samples.
        y_test (array-like): Target values for testing.
        models (dict): Dictionary of models to evaluate.
        params (dict): Dictionary of parameter grids for GridSearchCV.

    Returns:
        dict: Dictionary containing model names as keys and their R-squared scores as values.
    """

    X_train = args[0]
    X_test = args[1]
    y_train = args[2]
    y_test = args[3]

    report = {}

    gs = GridSearchCV(model, params, cv=3)
    gs.fit(X_train, y_train)

    model.set_params(**gs.best_params_)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    train_model_r2score = r2_score(y_train, y_train_pred)

    test_model_r2score = r2_score(y_test, y_test_pred)

    return train_model_r2score, test_model_r2score



def train_and_evaluate_model(model, *args):
    """
    Train and evaluate a model using the provided data.
    Specifically used for ANOVA test
    """
    X_train = args[0]
    X_test = args[1]
    y_train = args[2]
    y_test = args[3]

    model.fit(X_train, y_train)

    # Calculate the accuracy of the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return train_accuracy, test_accuracy
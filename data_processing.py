import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Returns column names that have the same value throughout (not useful for modeling).
def find_constant_columns(df):
    """Find columns that contain only a single value"""
    constant_columns = []
    for column in df.columns:
        if df[column].nunique() == 1:
            constant_columns.append(column)
    return constant_columns

#Finds columns with low variability (threshold defaults to 5).
def find_columns_with_few_values(df, threshold=5):
    """Find columns that have fewer unique values than the threshold"""
    columns_with_few_values = []
    for column in df.columns:
        if df[column].nunique() < threshold:
            columns_with_few_values.append(column)
    return columns_with_few_values

#Encodes categorical columns into binary features using pd.get_dummies.
def one_hot_encoding(df):
    """Perform one-hot encoding on categorical variables"""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns)
    return df

#Drops columns with more than 50% missing values.
# Fills missing values in:
# Numerical columns with mean
# Categorical columns with mode
def drop_and_fill(df):
    """Drop columns with too many missing values and fill remaining NaN values"""
    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)
    
    # Fill remaining numerical missing values with mean
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill remaining categorical missing values with mode
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

#Splits the dataset into training and test sets using train_test_split.
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    
    Args:
        df: pandas DataFrame containing the features and target
        target_column: name of the target column
        test_size: proportion of data to use for testing
        random_state: random seed for reproducibility
    
    Returns:
        x_train, x_test, y_train, y_test
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
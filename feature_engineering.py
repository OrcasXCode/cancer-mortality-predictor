import pandas as pd
import numpy as np

# bin_to_num: Converts a binned income category column (binnedInc) into numerical codes.
# cat_to_col: Converts all categorical columns to numerical codes.
# one_hot_encoding (repeated, same as in data_processing) – probably a redundancy to clean up.
def bin_to_num(df):
    """Convert binned income categories to numerical values"""
    # Using the correct column name 'binnedInc'
    df['binnedInc'] = df['binnedInc'].astype('category')
    df['binnedInc'] = df['binnedInc'].cat.codes
    return df

def cat_to_col(df):
    """Convert categorical columns to numerical"""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    return df

def one_hot_encoding(df):
    """Perform one-hot encoding on categorical variables"""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns)
    return df
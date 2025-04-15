import warnings
warnings.filterwarnings("ignore")

# Imports libraries for:
# Data manipulation (pandas, numpy)
# Plotting (matplotlib)
# Regression modeling (statsmodels)
# Local functions: split_data() for train-test splitting and IngestData to load data.
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from data_processing import split_data
from data_ingest import IngestData

# Computes the correlation matrix of selected columns.
# Identifies highly correlated column pairs (correlation > 0.8) to possibly drop them later (due to multicollinearity).
# Returns a set of such columns.
def correlation_among_numeric_features(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()
    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features


# Adds an intercept term (constant column).
# Builds a Linear Regression model using OLS (Ordinary Least Squares) from statsmodels.
# Fits and returns the trained model.
def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr


# Prints p-values and R² metrics from the model.
# Selects features with p-values less than 0.05 (statistically significant).
# Returns a list of significant variable names.
def identify_significant_vars(lr, p_value_threshold=0.05):
    print(lr.pvalues)
    print(lr.rsquared)
    print(lr.rsquared_adj)
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars


# Caps outliers in numeric columns.
# Any value beyond ±3 * standard deviation from the mean is clipped (limited) to that range.
def cap_outliers(df, columns, n_sigmas=3):
    df_capped = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df_capped[col] = df[col].clip(lower=mean - n_sigmas*std,
                                     upper=mean + n_sigmas*std)
    return df_capped

#Ensures this block only runs when executing this file directly
if __name__ == "__main__":
    # Load the original data
    #Loads the cancer dataset using a custom IngestData class.
    ingest_data = IngestData()
    df = ingest_data.get_data("data/cancer_reg.csv")
    
    # Cap outliers in numerical columns
    #Gets numeric columns and caps outliers.
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    capped_data = cap_outliers(df, numerical_columns)
    
    # Save capped data
    #Saves the cleaned data and prints its shape.
    capped_data.to_csv("data/capped_data.csv", index=False)
    print("Capped data shape:", capped_data.shape)

    # Define highly correlated columns to remove
    # Hardcoded list of features known to be highly correlated.
    # Possibly selected beforehand to avoid multicollinearity issues.
    highy_corr_cols = [
        "povertypercent",
        "median",
        "pctprivatecoveragealone",
        "medianagefemale",
        "pctempprivcoverage",
        "pctblack",
        "popest2015",
        "pctmarriedhouseholds",
        "upper_bound",
        "lower_bound",
        "pctprivatecoverage",
        "medianagemale",
        "state_District of Columbia",
        "pctpubliccoveragealone",
    ]

    # Get columns for modeling
    #Filters out those high-correlation features.
    cols = [col for col in capped_data.columns if col not in highy_corr_cols]
    print("Number of features:", len(cols))

    # Split data
    #Splits data into train/test sets with target as "TARGET_deathRate".
    x_train, x_test, y_train, y_test = split_data(capped_data[cols], "TARGET_deathRate")
    
    # Initial model
    # Trains the initial linear regression model and prints a full summary (coefficients, p-values, R², etc).
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary)

    # Get significant variables
    significant_vars = identify_significant_vars(lr)
    print("Number of significant variables:", len(significant_vars))

    # Train model with only significant variables
    #Removes the constant term from variable list (not needed in feature set).
    if "const" in significant_vars:
        significant_vars.remove("const")
    #Trains a refined model with only significant predictors.
    x_train = x_train[significant_vars]
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary)
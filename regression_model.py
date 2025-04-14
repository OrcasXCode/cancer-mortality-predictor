import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from data_processing import split_data
from data_ingest import IngestData

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

def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr, p_value_threshold=0.05):
    print(lr.pvalues)
    print(lr.rsquared)
    print(lr.rsquared_adj)
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars

def cap_outliers(df, columns, n_sigmas=3):
    df_capped = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df_capped[col] = df[col].clip(lower=mean - n_sigmas*std,
                                     upper=mean + n_sigmas*std)
    return df_capped

if __name__ == "__main__":
    # Load the original data
    ingest_data = IngestData()
    df = ingest_data.get_data("data/cancer_reg.csv")
    
    # Cap outliers in numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    capped_data = cap_outliers(df, numerical_columns)
    
    # Save capped data
    capped_data.to_csv("data/capped_data.csv", index=False)
    print("Capped data shape:", capped_data.shape)

    # Define highly correlated columns to remove
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
    cols = [col for col in capped_data.columns if col not in highy_corr_cols]
    print("Number of features:", len(cols))

    # Split data
    x_train, x_test, y_train, y_test = split_data(capped_data[cols], "TARGET_deathRate")
    
    # Initial model
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary)

    # Get significant variables
    significant_vars = identify_significant_vars(lr)
    print("Number of significant variables:", len(significant_vars))

    # Train model with only significant variables
    if "const" in significant_vars:
        significant_vars.remove("const")
    x_train = x_train[significant_vars]
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary)
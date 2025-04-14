# import plotly.graph_objects as go
# from data_ingest import IngestData

# # Load the data
# ingest_data = IngestData()
# df = ingest_data.get_data("data/cancer_reg.csv")

# # Create boxplot
# fig = go.Figure(data=[go.Box(
#     y=df['TARGET_deathRate'], 
#     boxpoints='outliers',
#     jitter=0.3,        # Added jitter parameter
#     pointpos=-1.8      # Added pointpos parameter
# )])

# # Update layout
# fig.update_layout(
#     title='Boxplot of Death Rate',
#     yaxis_title='Death Rate',
#     width=700,
#     height=500
# )

# fig.show()


# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from data_ingest import IngestData

# # Load the data
# ingest_data = IngestData()
# df = ingest_data.get_data("data/cancer_reg.csv")

# # Analyzing avgAnnCount column
# avg_ann_count_mean = df["avgAnnCount"].mean()
# avg_ann_count_std = df["avgAnnCount"].std()
# print("Mean of avgAnnCount: ", avg_ann_count_mean)
# print("Standard deviation of avgAnnCount: ", avg_ann_count_std)

# # Create matplotlib histogram
# plt.hist(df["avgAnnCount"], bins=20)
# plt.xlabel("avgAnnCount")
# plt.ylabel("Frequency")
# plt.title("Histogram of avgAnnCount")
# plt.show()

# # Create plotly boxplot for avgAnnCount
# fig = go.Figure(data=[go.Box(
#     y=df['avgAnnCount'],
#     boxpoints='outliers',
#     jitter=0.3,
#     pointpos=-1.8
# )])

# fig.update_layout(
#     title='Boxplot of AvgAnnCount',
#     yaxis_title='Count',
#     width=700,
#     height=500
# )
# fig.show()

# # Calculate correlation
# corr = df["avgAnnCount"].corr(df["TARGET_deathRate"])
# print("Correlation between avgAnnCount and TARGET_deathRate: ", corr)

# # Create scatter plot
# plt.scatter(df["avgAnnCount"], df["TARGET_deathRate"])
# plt.xlabel("avgAnnCount")
# plt.ylabel("target_deathrate")
# plt.title("Scatter plot of avgAnnCount vs target_deathrate")
# plt.show()


# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from data_ingest import IngestData
# from scipy.stats import normaltest
# import numpy as np

# # Load the data
# ingest_data = IngestData()
# df = ingest_data.get_data("data/cancer_reg.csv")

# # Analyzing avgAnnCount column
# avg_ann_count_mean = df["avgAnnCount"].mean()
# avg_ann_count_std = df["avgAnnCount"].std()
# print("Mean of avgAnnCount: ", avg_ann_count_mean)
# print("Standard deviation of avgAnnCount: ", avg_ann_count_std)

# # Create matplotlib histogram
# plt.hist(df["avgAnnCount"], bins=20)
# plt.xlabel("avgAnnCount")
# plt.ylabel("Frequency")
# plt.title("Histogram of avgAnnCount")
# plt.show()

# # Create plotly boxplot for avgAnnCount
# fig = go.Figure(data=[go.Box(
#     y=df['avgAnnCount'],
#     boxpoints='outliers',
#     jitter=0.3,
#     pointpos=-1.8
# )])

# fig.update_layout(
#     title='Boxplot of AvgAnnCount',
#     yaxis_title='Count',
#     width=700,
#     height=500
# )
# fig.show()

# # Calculate correlation
# corr = df["avgAnnCount"].corr(df["TARGET_deathRate"])
# print("Correlation between avgAnnCount and TARGET_deathRate: ", corr)

# # Create scatter plot
# plt.scatter(df["avgAnnCount"], df["TARGET_deathRate"])
# plt.xlabel("avgAnnCount")
# plt.ylabel("target_deathrate")
# plt.title("Scatter plot of avgAnnCount vs target_deathrate")
# plt.show()

# # Perform normality test on numerical columns
# numerical_columns = df.select_dtypes(include=np.number).columns
# gaussian_cols = []
# non_gaussian_cols = []
# for col in numerical_columns:
#     stat, p = normaltest(df[col])
#     print('Statistics=%.3f, p=%.3f' % (stat, p))
#     alpha = 0.05
#     if p > alpha:
#         gaussian_cols.append(col)
#     else:
#         non_gaussian_cols.append(col)
# print("Gaussian distributed columns:", gaussian_cols)

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from data_ingest import IngestData
from scipy.stats import normaltest
import numpy as np
import pandas as pd

def deal_with_outliers(df, col, basic_info):
    highest_allowed = basic_info[col]["mean"] + 3*basic_info[col]["std"]
    lowest_allowed = basic_info[col]["mean"] - 3*basic_info[col]["std"]
    df = df[~((df[col] > highest_allowed) | (df[col] < lowest_allowed))]
    return df

# Load the data
ingest_data = IngestData()
df = ingest_data.get_data("data/cancer_reg.csv")

# Perform normality test on numerical columns
numerical_columns = df.select_dtypes(include=np.number).columns
gaussian_cols = []
non_gaussian_cols = []
for col in numerical_columns:
    stat, p = normaltest(df[col])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        gaussian_cols.append(col)
    else:
        non_gaussian_cols.append(col)
print("Gaussian distributed columns:", gaussian_cols)

# Calculate basic info for Gaussian columns
basic_info_gaussian = {}
for col in gaussian_cols:
    basic_info_gaussian[col] = {
        "mean": df[col].mean(),
        "std": df[col].std()
    }

# Deal with outliers in Gaussian columns
cols_have_outliers = []
for col in gaussian_cols:
    df_cleaned = deal_with_outliers(df, col, basic_info_gaussian)
    shape = df_cleaned.shape
    if shape[0] > 0:
        cols_have_outliers.append(col)
print("Columns with outliers:", cols_have_outliers)

# Visualizations for a sample column (avgAnnCount)
plt.hist(df["avgAnnCount"], bins=20)
plt.xlabel("avgAnnCount")
plt.ylabel("Frequency")
plt.title("Histogram of avgAnnCount")
plt.show()

fig = go.Figure(data=[go.Box(
    y=df['avgAnnCount'],
    boxpoints='outliers',
    jitter=0.3,
    pointpos=-1.8
)])

fig.update_layout(
    title='Boxplot of AvgAnnCount',
    yaxis_title='Count',
    width=700,
    height=500
)
fig.show()

def identify_skewed_cols(df, cols):
    skewed_cols = []
    for col in cols:
        skew = df[col].skew()
        if skew > 1 or skew < -1:
            skewed_cols.append(col)
    return skewed_cols

# Remove columns with less than 10 unique values
cols_to_remove = []
for col in processed_data.columns:
    if processed_data[col].nunique() < 10:
        cols_to_remove.append(col)
print("Number of columns removed:", len(cols_to_remove))

data_for_skewness = processed_data.drop(cols_to_remove, axis=1)
skewed_cols = identify_skewed_cols(data_for_skewness, data_for_skewness.columns)
print("Number of skewed columns:", len(skewed_cols))
print("Skewed columns:", skewed_cols)
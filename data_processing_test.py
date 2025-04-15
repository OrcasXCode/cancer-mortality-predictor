from data_ingest import IngestData
from data_processing import (
    drop_and_fill,
    find_constant_columns,
    find_columns_with_few_values,
    one_hot_encoding
)
from feature_engineering import bin_to_num, cat_to_col,one_hot_encoding
import matplotlib.pyplot as plt


ingest_data = IngestData()
df = ingest_data.get_data("data/cancer_reg.csv")
constant_columns = find_constant_columns(df)
print("Columns that contain a single value: ", constant_columns)
columns_with_few_values = find_columns_with_few_values(df)

df["binnedInc"][0]  # Changed from binnedinc to binnedInc
df = bin_to_num(df)
df = cat_to_col(df)
df = one_hot_encoding(df)
df = drop_and_fill(df)
print(df.shape)
df.to_csv("data/cancer_reg.csv")

plt.hist(df['TARGET_deathRate'], bins=20, color='blue', edgecolor='black')  # Changed from target_deathrate to TARGET_deathRate
plt.xlabel('Death Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Death Rate')
plt.show()


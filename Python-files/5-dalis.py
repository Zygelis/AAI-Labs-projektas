import pandas as pd

# Read in data
df = pd.read_excel("WEOOct2020all.xls", engine="xlrd")

# TASK 5
# Find all the data fields from the year 2015 that are present in most of the countries.

# Count how many countries are in the dataset
countries = len(df["Country"].unique())

# Select most_countries threshold. I chose that most countries are 75% of the countries.
most_countries = countries * 0.75

# Group data by the WEO Subject Code and count how many countries have data in 2015
df_count = df.groupby("WEO Subject Code")[2015].count()
print(df_count)

# Filter dataframe to only include data fields that are present in most of the countries
df_most = df_count[df_count > most_countries]
print(df_most.index)

# In one line
df_most_1 = df.groupby("WEO Subject Code")[2015].count()[lambda x: x > most_countries]
print(df_most_1.index)

# Find Weo Subject Code descriptions
df_desc = df[["WEO Subject Code", "Subject Descriptor"]].drop_duplicates()
print(df_desc[df_desc["WEO Subject Code"].isin(df_most.index)])

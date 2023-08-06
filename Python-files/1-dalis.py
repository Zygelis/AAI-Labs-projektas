import pandas as pd

# Read in the data
df = pd.read_excel("WEOOct2020all.xls", engine="xlrd")

# 1 Task
# Find top 10 countries that grew "Gross domestic product per capita" the most over the last decade

# Choose decade years
year_end = df["Estimates Start After"].value_counts().index[1].astype(int)
year_start = year_end - 10

# Choose rows where "WEO Subject Code" is NGDPDPC
df = df[df["WEO Subject Code"] == "NGDPDPC"]

# Select only columns that are required for calculations
df = df[["Country", year_start, year_end]].dropna()

# Calculate GDP change over the decade
df["GDP_change"] = df[year_end] - df[year_start]

# Sort the countries based on their GDP per capita growth rates in descending order
df_res = df.sort_values(by="GDP_change", ascending=False)

print(df_res.head(10)[["Country", "GDP_change"]])
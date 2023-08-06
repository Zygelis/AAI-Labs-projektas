import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
df = pd.read_excel("WEOOct2020all.xls", engine="xlrd")

# 3 task:
# Save the GDP growth figures in separate charts and save them as PNG files

# Select GDP per capita data
df = df[(df["WEO Subject Code"] == "NGDPDPC")]


# Function to plot GDP growth figures of selected country
def plot_creation(df, country):
    # Select country data
    df = df[(df["Country"] == country)]

    # Set country name as index
    df = df.set_index("Country")

    # Some countries have 0 or nan in 'Estimates Start After' column, so we set it to 2020
    if (
        df["Estimates Start After"].isnull().values.any()
        or df["Estimates Start After"].iloc[0] == 0
    ):
        df["Estimates Start After"] = 2020

    # Determine decade start and end years for this country
    year_end = int(df["Estimates Start After"].iloc[0])
    year_start = year_end - 10

    # Select columns from start_year to end_year
    df = df.loc[:, year_start:year_end]

    # If there are nan values skip that country
    if df.isnull().values.any():
        print(f"{country} has nan values")
        return

    # Reset Seaborn settings to make sure we have a clean plot
    sns.set()

    # Plot GDP growth figures
    plot = sns.lineplot(data=df.loc[country, year_start:year_end], marker="o")
    sns.set_style("darkgrid")

    plot.set_title(f"{country} GDP growth over the last decade")
    plot.set_xlabel("Year")
    plot.set_ylabel("GDP")

    # Save the plot
    plot.figure.savefig(f"3-dalis\\Grafikai\\{country}_GDP_growth.png")

    # Close the plot to make separate charts for each country
    plt.close()


# Select all countries from dataframe
country_list = df["Country"].unique()

for country in country_list:
    plot_creation(df, country)

print("Done")

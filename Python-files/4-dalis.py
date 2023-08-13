import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_excel("C:\\Users\\Zygis\\Desktop\\test\\WEOOct2020all.xls", engine="xlrd")

# 4 task:
# Create 5 clusters out of the countries using GDP and "Volume of exports of goods"


# Function to clean the data
def clean_data(df, year):
    # Select rows where WEO Subject Code is NGDPD, TXG_RPCH
    df = df.loc[df["WEO Subject Code"].isin(["NGDPD", "TXG_RPCH"])]

    # Drop this column because it causes problems when reshaping data
    df.drop("Country/Series-specific Notes", axis=1)

    # Select only country and year
    df = df[["Country", year]]

    # Clean strange values that could corrupt data
    df = (
        df.replace("\n", np.nan)
        .replace("\t", np.nan)
        .replace(";", np.nan)
        .replace("--", np.nan)
    )

    # Remove countries with any missing values
    df = df.groupby("Country").filter(lambda x: not x.isnull().any().any())

    # Use pivot_table to reshape data
    df = df.pivot_table(
        index="Country",
        columns=df.groupby("Country").cumcount(),
        values=year,
        aggfunc="sum",
    ).reset_index()

    # Rename the columns
    df.columns = ["Country", "NGDPD", "TXG_RPCH"]

    # Convert column NGDPD to numeric
    df["NGDPD"] = pd.to_numeric(df["NGDPD"])

    # Convert column TXG_RPCH to numeric
    df["TXG_RPCH"] = pd.to_numeric(df["TXG_RPCH"])

    # Scale the data
    scaled_df = StandardScaler().fit_transform(df[["NGDPD", "TXG_RPCH"]])

    scaled_df = pd.DataFrame(scaled_df, columns=["NGDPD", "TXG_RPCH"])

    # Or keep the original data
    df_target_original = df[["NGDPD", "TXG_RPCH"]]

    df = df[["Country", "NGDPD", "TXG_RPCH"]]

    return df, scaled_df, df_target_original


def apply_kmeans(df, df_to_cluster, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=15, random_state=0)
    df["Cluster"] = kmeans.fit_predict(df_to_cluster)

    return df


def plot_clusters(df):
    plt.figure(figsize=(15, 10))
    sns.scatterplot(
        data=df, x="NGDPD", y="TXG_RPCH", hue="Cluster", palette="Set1", s=55
    )
    plt.title("K-Means Clustering of Countries based on GDP and Volume of Exports")
    plt.xlabel("GDP")
    plt.ylabel("Volume of exports of goods")
    plt.legend(title="Cluster")
    plt.grid(True)

    # Calculate top 5 countries per cluster
    top_countries_per_cluster = df.groupby("Cluster").apply(
        lambda x: x.nlargest(5, "NGDPD")
    )

    print("Top 5 countries per cluster:")
    print(top_countries_per_cluster)
    print("\n")

    # Annotate the top 5 countries per cluster
    for ids, row in top_countries_per_cluster.iterrows():
        plt.annotate(
            row["Country"],
            (row["NGDPD"], row["TXG_RPCH"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    plt.show()


# Clean the data
df, scaled_df, df_target_original = clean_data(df, 2018)

# Apply k-means clustering with 5 clusters
df = apply_kmeans(df, scaled_df, 5)

# Visualize the clusters
plot_clusters(df)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
df = pd.read_excel("C:\\Users\\Zygis\\Desktop\\test\\WEOOct2020all.xls", engine="xlrd")

# Features that are not related to GDP per capita, except 'NGDPDPC' which is the target variable.
all_features = [
    "NGDPDPC",
    "PPPEX",
    "PCPI",
    "PCPIE",
    "TM_RPCH",
    "TMG_RPCH",
    "TX_RPCH",
    "TXG_RPCH",
    "LUR",
    "LE",
    "LP",
    "GGR",
    "GGX",
    "GGXCNL",
    "GGSB",
    "GGXONLB",
    "GGXWDN",
    "GGXWDG",
    "BCA",
]


# Data preparation
def data_preparation(df, features):
    df = df.drop(
        [
            "Country",
            "Subject Descriptor",
            "Units",
            "Subject Notes",
            "Country/Series-specific Notes",
            "Estimates Start After",
            "Scale",
            "ISO",
        ],
        axis=1,
    )

    # Clean strange values that could corrupt data
    df = (
        df.replace("\n", np.nan)
        .replace("\t", np.nan)
        .replace(";", np.nan)
        .replace("--", np.nan)
    )

    # Select rows where WEO Subject Code is features
    df = df[df["WEO Subject Code"].isin(features)]

    # MODIFYING DATAFRAME STRUCTURE
    # Use melt to move years to a single column
    df_melted = df.melt(
        id_vars=["WEO Country Code", "WEO Subject Code"],
        var_name="Year",
        value_name="Value",
    )

    # Use pivot to make each feature a column
    df_pivoted = df_melted.pivot_table(
        index=["WEO Country Code", "Year"],
        columns="WEO Subject Code",
        values="Value",
        aggfunc="first",
    )

    # Reset the index to return index to columns
    df_pivoted.reset_index(inplace=True)

    # DEALING WITH NAN VALUES
    # Drop column if more then 40% values are nan
    df_pivoted = df_pivoted.dropna(axis=1, thresh=int(len(df_pivoted) * 0.5))

    # Drop row if more then 40% values are nan
    df_pivoted = df_pivoted.dropna(axis=0, thresh=int(len(df_pivoted.columns) * 0.5))

    # Fill nan values with mean of collumn
    df_pivoted = df_pivoted.fillna(df_pivoted.mean())

    # Now we have clean data
    # Drop "WEO Country Code", "Year" because they are not needed
    df_pivoted = df_pivoted.drop(["WEO Country Code", "Year"], axis=1)

    # To excel
    df_pivoted.to_excel("C:\\Users\\Zygis\\Desktop\\test\\df_cleaned.xlsx")

    # Split data to X and y
    df_X = df_pivoted.drop(["NGDPDPC"], axis=1)

    df_y = df_pivoted["NGDPDPC"]

    return df_X, df_y


def data_split(df_X, df_y):
    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

    return X_train, X_test, y_train, y_test


def data_scale(df_X):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_X)

    return X_scaled


# Feature selection (mutual information)
def make_mi_scores(X, y):
    # Create discrete features for mutual information calculation
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(mi_scores):
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=mi_scores, y=mi_scores.index)
    ax.set_title("Mutual Information Scores")
    plt.show()


df_X, df_y = data_preparation(df, all_features)

mi_scores = make_mi_scores(df_X, df_y)

X_train, X_test, y_train, y_test = data_split(df_X, df_y)

print("6_Data_preparation.py done")

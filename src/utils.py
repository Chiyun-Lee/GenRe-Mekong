import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove":
                 ["zoom", "pan", "zoomin", "zoomout", "lasso", "pan", "select2d", "resetscale", "toImage"]}

def load_df():
    """
    Loads the data from local .xlsx files and does some very simple cleaning
    """
    df = pd.read_excel("src/20200705-GenRe-PfMasterData-0.39.xlsx", sheet_name = "GenRe-Mekong", engine = "openpyxl")

    df = df.loc[
                (df["Bcode\nValid"] == True) &
                (df.Species != "-")
                # & (df["Admission Sample"] == False)
                ].drop(columns = ["SeqNum", "Pf sample"])

    return df

def find_location_details(s):
    """
    Uses geopy package to query an API to return longitude, latitude and country information of District/City names
    """
    if "geolocator" not in locals():
        geolocator = Nominatim(user_agent = "MyApp")
    
    location = geolocator.geocode(s, language = "en")

    try:
        return [location.latitude, location.longitude, location.address.split(", ")[-1]]
    
    except AttributeError:
        return [None, None, None]


def generate_sparsity_plots(df, n_bins = 10):
    """
    Plots for analysing the prevalence of "X"s in each row and each column
    """
    fig, axes = plt.subplots(2, 2, figsize = (7, 3))

    axes[0][0].sharex(axes[1][0])
    axes[0][0].set_title("Sparsity per ROW")
    axes[1][0].set_xlabel("% sparsity")
    axes[0][0].hist([100*(row == "X").sum() / len(row) for row in df.values], bins = n_bins)
    axes[1][0].hist([100*(row == "X").sum() / len(row) for row in df.values], bins = n_bins, log = "y")

    axes[0][1].sharex(axes[1][1])
    axes[0][1].set_title("Sparsity per COL")
    axes[1][1].set_xlabel("% sparsity")
    axes[0][1].hist([100*df[col].value_counts()["X"] / df[col].value_counts().sum() for col in df.columns], bins = n_bins)
    axes[1][1].hist([100*df[col].value_counts()["X"] / df[col].value_counts().sum() for col in df.columns], bins = n_bins, log = "y")

    
    fig.text(x = 0.5, y = 1, s = f"{df.shape[0]} rows, {df.shape[1]} columns", ha = "center")
    fig.text(x = 0, y = 0.5, s = f"n rows (with/without log)", ha = "center", va = "center", rotation = 90)
    fig.text(x = 0.55, y = 0.5, s = f"n columns (with/without log)", ha = "center", va = "center", rotation = 90)
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.5)
    plt.show()

base_encoder = lambda x: {
    # Quick and dirty function for pseudo-one-hot-encoding the sequence info
    "A": np.array([1.0, 0.0, 0.0, 0.0]),
    "T": np.array([0.0, 1.0, 0.0, 0.0]),
    "C": np.array([0.0, 0.0, 1.0, 0.0]),
    "G": np.array([0.0, 0.0, 0.0, 1.0]),
    "X": np.array([0.2, 0.2, 0.2, 0.2]),
    "N": np.array([0.7, 0.7, 0.7, 0.7])}[x]

def optimal_dbscan_fit(encoded_df, min_samples = 5, test_eps_values = np.arange(0.3, 10, 0.2)):
    """
    A function that takes a list of epsilon values to try fitting a DBSCAN clustering algorithm with.
    Provides a dataframe of different metrics for each epsilon value to guide the selection of the optimal hyperparameter. 
    """
    test_eps_columns = [f"eps={i:.01f}" for i in test_eps_values]
    genotype_columns = encoded_df.columns
    l_labels = []

    for eps in test_eps_values:
        dbscan_clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(encoded_df.values)

        l_labels += [dbscan_clustering.labels_]

    eps_df = pd.DataFrame(l_labels).T
    eps_df.columns = test_eps_columns
    eps_df = pd.concat([encoded_df.reset_index(), eps_df], axis = 1).set_index("SampleId")
    
    # metrics described in more detail in main.ipynb
    l_percentage_minus_1 = []
    l_n_clusters = []
    l_errors = []
    for eps_col in test_eps_columns:
        eps_col_diffs = []
        l_percentage_minus_1 += [len(eps_df.loc[eps_df[eps_col] == -1]) / len(eps_df)]

        for cluster in eps_df[eps_col].unique():
            if cluster == -1:
                continue

            in_cluster_df = eps_df.loc[eps_df[eps_col] == cluster, genotype_columns]
            in_cluster_means = in_cluster_df.mean(axis = 0).to_numpy()

            out_cluster_df = eps_df.loc[eps_df[eps_col] != cluster, genotype_columns]
            out_closest_to_in_mean = np.min(out_cluster_df.subtract(in_cluster_means).abs().sum(axis = 1) / len(out_cluster_df))

            eps_col_diffs += [out_closest_to_in_mean]
        
        l_n_clusters += [eps_df[eps_col].nunique()]
        l_errors += [np.sum(eps_col_diffs)]
    

    return pd.DataFrame({"epsilon": test_eps_values,
                         "ratio cluster -1": l_percentage_minus_1, 
                         "n clusters": l_n_clusters,
                         "cluster uniqueness": l_errors}).dropna()



def spiral_offset(k, r_scale = 0.1, theta_scale = 0.6):
    """
    Mathematical function that helps distribute points in a spiral. 
    Helpful for when multiple scatter points are stacked directly on top of each other to help with visibility. 
    """
    offsets = []
    for i in range(k):
        theta = i * np.pi / 2
        radius = np.sqrt(i) * r_scale
        x = radius * np.cos(theta * theta_scale)
        y = radius * np.sin(theta * theta_scale)
        offsets += [[x, y]]
    return np.array(offsets).T


def generate_cluster_temporal_df(pca_df):
    """
    Cleaning function to overcome the quirks of plotly. Generates the data needed for plotting functions. 
    """
    fig_df = pca_df.groupby(["Year", "cluster", "AdmDiv1"]).apply(lambda x: pd.Series({"Count": len(x),
                                                                                       "Longitude": x.longitude.values[0],
                                                                                       "Latitude": x.latitude.values[0],
                                                                                       "District": x.AdmDiv1.values[0],
                                                                                       "Country": x.country.values[0],
                                                                                       "cluster_str": x.cluster_str.values[0],
                                                                                       })).reset_index()

    # the following steps involving the "buffer" are in place to overcome the quirks of plotly.
    # without these steps, the legend interacts strangely with the slider
    buffer = [pd.DataFrame({"Year": Year,
                            "cluster": cluster,
                            "cluster_str": f"Cluster {cluster}",
                            "Count": 0,
                            "Longitude": np.nan,
                            "Latitude": np.nan,
                            "District": 0,
                            "Country": 0}, index = [0]) for cluster in fig_df.cluster.unique() for Year in fig_df.Year.unique()]
    fig_df = pd.concat([fig_df] + buffer).sort_values(["Year", "cluster"]).reset_index(drop = True)
    fig_df["Size"] = fig_df.Count.apply(lambda x: 3 + 5*np.log(x+1))

    # the following steps helps to redistribute datapoints of the same city and spreads them out a bit for visibility
    jitter_df = fig_df.groupby(["Year", "District"]).apply(lambda x: pd.Series({"cluster": list(x.cluster),
                                                                            "idx": list(x.index),
                                                                            "Longitude": np.array(x.Longitude),
                                                                            "Latitude": np.array(x.Latitude),
                                                                            })).reset_index()

    jitter_df = jitter_df.loc[jitter_df.District != 0].reset_index(drop = True)
    jitter_df.Longitude = jitter_df.Longitude.apply(lambda x: x + spiral_offset(len(x))[0])
    jitter_df.Latitude = jitter_df.Latitude.apply(lambda x: x + spiral_offset(len(x))[1])

    for i in jitter_df.index:
        fig_df.loc[jitter_df.loc[i, "idx"], "Longitude"] = jitter_df.loc[i, "Longitude"]
        fig_df.loc[jitter_df.loc[i, "idx"], "Latitude"] = jitter_df.loc[i, "Latitude"]

    return fig_df
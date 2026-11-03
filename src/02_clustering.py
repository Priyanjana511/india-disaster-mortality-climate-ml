
# ==========================================================
# PROJECT: Disaster Death Analysis in India
# FILE: 02_clustering.py
# PURPOSE: State-wise Disaster Mortality Clustering
# ==========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram


# ----------------------------------------------------------
# 1. PROJECT PATHS
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 2. LOAD CLEANED DEATH RATE DATA
# ----------------------------------------------------------

def load_data():

    file = TABLES / "state_year_death_rates_2001_2014.csv"

    if not file.exists():
        raise FileNotFoundError(
            f"Death-rate dataset not found: {file}\n"
            "Run 01_data_preprocessing.py first."
        )

    df = pd.read_csv(file)

    required_columns = [
        "state",
        "lightning_rate_per_lakh",
        "heat_stroke_rate_per_lakh",
        "flood_rate_per_lakh"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {missing}"
        )

    print("\nDataset loaded successfully!")
    print("Rows:", len(df))
    print(df.head())

    return df


# ----------------------------------------------------------
# 3. CREATE STATE-WISE PROFILES
# ----------------------------------------------------------

def prepare_features(df):

    features = [
        "lightning_rate_per_lakh",
        "heat_stroke_rate_per_lakh",
        "flood_rate_per_lakh"
    ]

    # Average rates over 2001-2014 for each State.
    state_data = (
        df.groupby("state")[features]
        .mean()
        .reset_index()
    )

    state_data = state_data.replace(
        [np.inf, -np.inf], np.nan
    )

    state_data = state_data.dropna(
        subset=features
    )

    if len(state_data) < 3:
        raise ValueError(
            "At least three valid States are required "
            "for clustering analysis."
        )

    print("\nState-wise average mortality profiles:")
    print(state_data.to_string(index=False))

    print(
        "\nNumber of States/UTs included:",
        len(state_data)
    )

    X = state_data[features]

    return state_data, X


# ----------------------------------------------------------
# 4. STATE-WISE DEATH RATE HEATMAP
# ----------------------------------------------------------

def create_state_rate_heatmap(state_data):

    print("\nCreating State-wise death rate heatmap...")

    features = [
        "lightning_rate_per_lakh",
        "heat_stroke_rate_per_lakh",
        "flood_rate_per_lakh"
    ]

    heatmap_data = state_data[
        ["state"] + features
    ].copy()

    heatmap_data = heatmap_data.rename(
        columns={
            "lightning_rate_per_lakh": "Lightning",
            "heat_stroke_rate_per_lakh": "Heat Stroke",
            "flood_rate_per_lakh": "Floods"
        }
    )

    heatmap_data = heatmap_data.set_index("state")
    heatmap_data = heatmap_data.sort_index()

    fig_height = max(
        8,
        len(heatmap_data) * 0.38
    )

    plt.figure(
        figsize=(10, fig_height)
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Average annual deaths per lakh population"
        }
    )

    plt.title(
        "State-wise Natural Disaster Death Rates (2001-2014)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.xlabel("Cause of Death", fontsize=11)
    plt.ylabel("State / Union Territory", fontsize=11)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURES / "01_state_rate_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: 01_state_rate_heatmap.png")


# ----------------------------------------------------------
# 5. FEATURE SCALING
# ----------------------------------------------------------

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nFeature scaling completed.")

    return X_scaled


# ----------------------------------------------------------
# 6. ELBOW METHOD AND SILHOUETTE SCORE
# ----------------------------------------------------------

def evaluate_clusters(X_scaled):

    results = []

    max_k = min(8, len(X_scaled) - 1)

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        results.append({
            "k": k,
            "SSE": model.inertia_,
            "Silhouette": silhouette
        })

    if not results:
        raise ValueError(
            "Not enough States to evaluate clusters."
        )

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        TABLES / "clustering_metrics.csv",
        index=False
    )

    print("\nClustering evaluation:")
    print(metrics.to_string(index=False))

    # Elbow graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["k"],
        metrics["SSE"],
        marker="o"
    )

    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES / "02_elbow_sse.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Silhouette graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["k"],
        metrics["Silhouette"],
        marker="o"
    )

    plt.title("Silhouette Score Analysis")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES / "03_silhouette_scores.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Select highest silhouette score as candidate K
    best_k = int(
        metrics.loc[
            metrics["Silhouette"].idxmax(),
            "k"
        ]
    )

    print("\nBest candidate K:", best_k)

    return best_k


# ----------------------------------------------------------
# 7. K-MEANS CLUSTERING
# ----------------------------------------------------------

def perform_kmeans(state_data, X_scaled, best_k):

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )

    state_data["Cluster"] = model.fit_predict(
        X_scaled
    )

    print("\nK-Means clustering completed.")

    print(
        state_data[
            ["state", "Cluster"]
        ].to_string(index=False)
    )

    return state_data


# ----------------------------------------------------------
# 8. PCA VISUALIZATION
# ----------------------------------------------------------

def perform_pca(state_data, X_scaled):

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    state_data["PC1"] = components[:, 0]
    state_data["PC2"] = components[:, 1]

    print(
        "\nExplained variance:",
        pca.explained_variance_ratio_
    )

    print(
        "Total variance explained:",
        pca.explained_variance_ratio_.sum()
    )

    plt.figure(figsize=(11, 7))

    sns.scatterplot(
        data=state_data,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=100
    )

    for _, row in state_data.iterrows():

        plt.annotate(
            row["state"],
            (row["PC1"], row["PC2"]),
            fontsize=7
        )

    plt.title(
        "PCA Visualization of Disaster Clusters"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()

    plt.savefig(
        FIGURES / "04_pca_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return state_data


# ----------------------------------------------------------
# 9. HIERARCHICAL CLUSTERING
# ----------------------------------------------------------

def hierarchical_clustering(state_data, X_scaled):

    Z = linkage(
        X_scaled,
        method="ward"
    )

    plt.figure(figsize=(14, 7))

    dendrogram(
        Z,
        labels=state_data["state"].tolist(),
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title(
        "Hierarchical Clustering Dendrogram"
    )

    plt.xlabel("States and Union Territories")
    plt.ylabel("Ward Distance")

    plt.tight_layout()

    plt.savefig(
        FIGURES / "05_hierarchical_dendrogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nHierarchical clustering completed.")


# ----------------------------------------------------------
# 10. OUTLIER DETECTION
# ----------------------------------------------------------

def detect_outliers(state_data, X_scaled):

    model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)

    state_data["Outlier"] = predictions

    state_data["Outlier_Status"] = (
        state_data["Outlier"].map({
            1: "Normal",
            -1: "Potential Outlier"
        })
    )

    print("\nOutlier detection completed.")

    print(
        state_data[
            ["state", "Outlier_Status"]
        ].to_string(index=False)
    )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=state_data,
        x="lightning_rate_per_lakh",
        y="heat_stroke_rate_per_lakh",
        hue="Outlier_Status",
        style="Outlier_Status",
        s=100
    )

    plt.title(
        "Outlier Detection in Disaster Mortality"
    )

    plt.xlabel("Lightning Death Rate per Lakh")
    plt.ylabel("Heat Stroke Death Rate per Lakh")

    plt.tight_layout()

    plt.savefig(
        FIGURES / "06_outlier_detection.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return state_data


# ----------------------------------------------------------
# 11. SAVE FINAL RESULTS
# ----------------------------------------------------------

def save_results(state_data):

    file = TABLES / "final_clustering_results.csv"

    state_data.to_csv(
        file,
        index=False
    )

    print("\nFinal clustering results saved!")
    print(file)


# ----------------------------------------------------------
# 12. MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("STARTING DISASTER MORTALITY CLUSTERING ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()

    # Prepare State-wise profiles
    state_data, X = prepare_features(df)

    # Graph 1: State-wise mortality heatmap
    create_state_rate_heatmap(state_data)

    # Standardise features
    X_scaled = scale_features(X)

    # Graphs 2 and 3: Cluster evaluation
    best_k = evaluate_clusters(X_scaled)

    # K-Means clustering
    state_data = perform_kmeans(
        state_data,
        X_scaled,
        best_k
    )

    # Graph 4: PCA visualization
    state_data = perform_pca(
        state_data,
        X_scaled
    )

    # Graph 5: Hierarchical dendrogram
    hierarchical_clustering(
        state_data,
        X_scaled
    )

    # Graph 6: Outlier detection
    state_data = detect_outliers(
        state_data,
        X_scaled
    )

    # Save final results
    save_results(state_data)

    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS COMPLETED!")
    print("All six graphs have been generated.")
    print("CSV files saved inside outputs/tables/")
    print("Graphs saved inside outputs/figures/")
    print("=" * 60)


# ----------------------------------------------------------
# 13. EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":
    main()

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parents[1]  # script should be in project/src/
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)
sns.set_theme(style="whitegrid")


def norm_state(value):
    s = re.sub(r"\s+", " ", str(value).strip().upper()).replace("&", "AND")
    aliases = {
        "A AND N ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "D AND N HAVELI": "DADRA AND NAGAR HAVELI",
        "DANDN HAVELI": "DADRA AND NAGAR HAVELI",
        "DELHI (UT)": "DELHI", "DELHI UT": "DELHI", "NCT OF DELHI": "DELHI",
        "ORISSA": "ODISHA", "UTTARANCHAL": "UTTARAKHAND",
        # Telangana was part of Andhra Pradesh in the 2011 Census.
        "TELANGANA": "ANDHRA PRADESH",
    }
    return aliases.get(s, s)


def norm_cause(value):
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    if s == "total" or s.startswith("total "):
        return None
    if "lightning" in s:
        return "lightning"
    if "heat" in s and ("stroke" in s or "sun" in s):
        return "heat_stroke"
    if "flood" in s:
        return "flood"
    if "other natural" in s:
        return "other_natural"
    return None


def read_csv(path):
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path.name}")


def standardise_file(path):
    df = read_csv(path)
    if {"STATE/UT", "CAUSE"}.issubset(df.columns):
        state_col, cause_col = "STATE/UT", "CAUSE"
    elif {"States/UTs", "CrimeHead"}.issubset(df.columns):
        state_col, cause_col = "States/UTs", "CrimeHead"
    else:
        raise ValueError(f"Unexpected columns in {path.name}: {df.columns.tolist()}")
    required = [state_col, "Year", cause_col, "Grand Total"]
    d = df[required].copy()
    d.columns = ["state_raw", "year", "cause_raw", "deaths"]
    d["state"] = d.state_raw.map(norm_state)
    d["cause_group"] = d.cause_raw.map(norm_cause)
    d["year"] = pd.to_numeric(d.year, errors="coerce")
    d["deaths"] = pd.to_numeric(d.deaths, errors="coerce")
    aggregate = {"TOTAL (ALL INDIA)", "TOTAL (STATES)", "TOTAL (UTS)", "INDIA"}
    d = d[~d.state_raw.astype(str).str.strip().str.upper().isin(aggregate)]
    d = d.dropna(subset=["state", "year", "deaths", "cause_group"])
    d = d[d.deaths >= 0].copy()
    d["year"] = d.year.astype(int)
    d = d[d.cause_group.isin(["lightning", "heat_stroke", "flood", "other_natural"])]
    return d[["state", "year", "cause_group", "deaths"]]


def locate(pattern, description):
    found = list(RAW.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description} matching {pattern}; found {[p.name for p in found]}")
    return found[0]


def load_deaths():
    # Supports both the original uploaded filenames and the shorter names
    # suggested in the project setup instructions.
    candidates = list(RAW.glob("*.csv"))
    f1_candidates = [p for p in candidates if "2001_2012" in p.name.lower()
                     or "2001-2012" in p.name.lower()
                     or p.name.lower().endswith("natural_causes_state(2).csv")]
    f2_candidates = [p for p in candidates if "2013" in p.name]
    f3_candidates = [p for p in candidates if "2014" in p.name]
    for label, found in [("2001-2012", f1_candidates), ("2013", f2_candidates), ("2014", f3_candidates)]:
        if len(found) != 1:
            raise FileNotFoundError(f"Expected exactly one {label} death CSV; found: {[p.name for p in found]}")
    f1, f2, f3 = f1_candidates[0], f2_candidates[0], f3_candidates[0]
    parts = []
    for f in (f1, f2, f3):
        print("Reading", f.name)
        parts.append(standardise_file(f))
    d = pd.concat(parts, ignore_index=True)
    d = d.groupby(["state", "year", "cause_group"], as_index=False).deaths.sum()
    d.to_csv(OUT / "adsi_cleaned.csv", index=False)
    return d


def load_population():
    xlsx = locate("*.xlsx", "Census workbook")
    c = pd.read_excel(xlsx, sheet_name="Data")
    required = {"Level", "Name", "TRU", "TOT_P"}
    if not required.issubset(c.columns):
        raise ValueError(f"Census columns missing: {required - set(c.columns)}")
    p = c[c.Level.astype(str).str.upper().eq("STATE") & c.TRU.astype(str).str.strip().str.lower().eq("total")].copy()
    p["state"] = p.Name.map(norm_state)
    p["population_2011"] = pd.to_numeric(p.TOT_P, errors="coerce")
    p = p[["state", "population_2011"]].dropna().drop_duplicates("state")
    p.to_csv(OUT / "state_population_2011.csv", index=False)
    return p


def make_rates(deaths, pop):
    w = deaths.pivot_table(index=["state", "year"], columns="cause_group", values="deaths", aggfunc="sum", fill_value=0).reset_index()
    w.columns.name = None
    for cause in ["lightning", "heat_stroke", "flood", "other_natural"]:
        if cause not in w.columns:
            w[cause] = 0
    w = w.merge(pop, on="state", how="left")
    missing = sorted(w.loc[w.population_2011.isna(), "state"].unique())
    if missing:
        print("WARNING: unmatched population State names:", missing)
    for cause in ["lightning", "heat_stroke", "flood", "other_natural"]:
        w[cause + "_rate_per_lakh"] = w[cause] / w.population_2011 * 100000
    w["population_note"] = "2011 Census population used as fixed denominator for 2001-2014; replace with annual estimates for final study."
    w.to_csv(OUT / "state_year_death_rates_2001_2014.csv", index=False)
    return w


def clustering(rates):
    features = ["lightning_rate_per_lakh", "heat_stroke_rate_per_lakh", "flood_rate_per_lakh"]
    profile = rates.groupby("state", as_index=False)[features].mean().replace([np.inf, -np.inf], np.nan).dropna()
    if len(profile) < 4:
        raise ValueError("Too few States with valid rates. Check population matching warnings.")
    X = StandardScaler().fit_transform(profile[features])

    # Compare candidate cluster counts. Inspect both SSE elbow and silhouette before final interpretation.
    metric_rows = []
    for k in range(2, min(9, len(profile))):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        metric_rows.append({"k": k, "sse": km.inertia_, "silhouette": silhouette_score(X, labels)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "clustering_metrics.csv", index=False)
    best_k = int(metrics.loc[metrics.silhouette.idxmax(), "k"])
    print("\nCluster metrics:\n", metrics.to_string(index=False))
    print("Candidate k with highest silhouette:", best_k)

    profile["kmeans_cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X)
    profile["hierarchical_cluster"] = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    flags = IsolationForest(random_state=42, contamination="auto").fit_predict(X)
    profile["outlier_status"] = np.where(flags == -1, "Review unusual profile", "Typical profile")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    profile["PC1"], profile["PC2"] = pcs[:, 0], pcs[:, 1]
    profile.to_csv(OUT / "state_clustering_results.csv", index=False)

    # Graph 1: Hazard rate heatmap
    plt.figure(figsize=(11, max(7, len(profile) * 0.27)))
    sns.heatmap(profile.set_index("state")[features], cmap="YlOrRd", linewidths=.2)
    plt.title("Average State-wise hazard death rates, 2001-2014")
    plt.xlabel("Deaths per lakh (2011 Census denominator)"); plt.ylabel("State / UT")
    plt.tight_layout(); plt.savefig(FIG / "01_state_hazard_rate_heatmap.png", dpi=250); plt.close()

    # Graph 2: SSE / elbow
    plt.figure(figsize=(8, 5)); plt.plot(metrics.k, metrics.sse, marker="o")
    plt.title("K-Means Elbow Method"); plt.xlabel("Number of clusters (k)"); plt.ylabel("SSE / Inertia")
    plt.xticks(metrics.k); plt.tight_layout(); plt.savefig(FIG / "02_kmeans_elbow_sse.png", dpi=250); plt.close()

    # Graph 3: Silhouette
    plt.figure(figsize=(8, 5)); plt.plot(metrics.k, metrics.silhouette, marker="o")
    plt.title("Silhouette Scores"); plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score")
    plt.xticks(metrics.k); plt.tight_layout(); plt.savefig(FIG / "03_silhouette_scores.png", dpi=250); plt.close()

    # Graph 4: PCA projection with K-Means labels
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=profile, x="PC1", y="PC2", hue="kmeans_cluster", palette="tab10", s=85)
    for _, r in profile.iterrows():
        plt.annotate(r.state, (r.PC1, r.PC2), xytext=(4, 3), textcoords="offset points", fontsize=7)
    plt.title("PCA view of State disaster-death profiles")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.tight_layout(); plt.savefig(FIG / "04_pca_kmeans_clusters.png", dpi=250); plt.close()

    # Graph 5: Hierarchical dendrogram
    Z = linkage(X, method="ward")
    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=profile.state.tolist(), leaf_rotation=90, leaf_font_size=7)
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
    plt.xlabel("State / UT"); plt.ylabel("Ward distance")
    plt.tight_layout(); plt.savefig(FIG / "05_hierarchical_dendrogram.png", dpi=250); plt.close()

    # Graph 6: Outlier screening
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=profile, x=features[0], y=features[1], hue="outlier_status", style="outlier_status", s=90)
    for _, r in profile.iterrows():
        plt.annotate(r.state, (r[features[0]], r[features[1]]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    plt.title("Outlier Screening: Lightning vs Heat-stroke death rates")
    plt.xlabel("Lightning deaths per lakh"); plt.ylabel("Heat-stroke deaths per lakh")
    plt.tight_layout(); plt.savefig(FIG / "06_outlier_screening.png", dpi=250); plt.close()

    print("\nSaved state clustering results and six graphs.")
    print("Cluster membership counts:\n", profile.kmeans_cluster.value_counts().sort_index())


def main():
    print("1. Cleaning ADSI death records...")
    deaths = load_deaths()
    print("Cleaned records:", len(deaths))
    print("2. Reading Census 2011 State population...")
    pop = load_population()
    print("Population records:", len(pop))
    print("3. Calculating death rates per lakh...")
    rates = make_rates(deaths, pop)
    print("State-year rate rows:", len(rates))
    print("4. Running scaling, PCA, K-Means, hierarchical clustering and outlier detection...")
    clustering(rates)
    print("DONE. Check outputs/ and outputs/figures/.")
    print("Research note: 2011 population is used as a fixed denominator for all years. Replace with annual population estimates for final results.")

if __name__ == "__main__":
    main()

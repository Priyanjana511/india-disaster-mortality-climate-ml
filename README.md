# Deaths from Lightning, Heat Stroke and Floods: A State Typology for Early-Warning Priorities

**Machine Learning | Disaster Management | Climate Analysis | India**

## 1. Project Overview

This project analyses state-wise accidental deaths caused by natural events in India, focusing on lightning, heat stroke, and floods.

The main objective is to identify patterns in disaster-related mortality across Indian States and Union Territories and explore the relationship between summer maximum temperature and heat-stroke mortality.

Machine Learning techniques are used to group States according to their disaster mortality profiles and to develop a regression model for analysing the association between temperature and heat-stroke deaths.

The findings aim to support data-driven disaster management and State-specific early-warning priorities.

## 2. Problem Statement

Deaths caused by lightning, heat waves and floods vary considerably across Indian States.

Understanding these differences can help identify regions that may require greater attention in disaster preparedness and early-warning systems.

This project addresses the following questions:

- How do lightning, heat-stroke and flood death rates vary across Indian States?
- Can States be grouped according to their natural-disaster mortality patterns?
- Which States show unusual mortality profiles?
- Is summer maximum temperature associated with heat-stroke mortality?

## 3. Sustainable Development Goals (SDGs)

This project contributes to the following United Nations Sustainable Development Goals:

- **SDG 13 – Climate Action:** Understanding climate-related risks and supporting climate adaptation.
- **SDG 11 – Sustainable Cities and Communities:** Supporting disaster-resilient communities.
- **SDG 3 – Good Health and Well-being:** Understanding mortality patterns associated with natural hazards.

## 4. Datasets

### A. Accidental Deaths in India

**Source:** Open Government Data (OGD) Platform India / National Crime Records Bureau (NCRB).

Dataset: State/UT-wise Distribution of Accidental Deaths by Natural Causes.

Study period: 2001–2014.

Variables include State/UT, year, cause of death and reported deaths.

Dataset link:
https://www.data.gov.in/catalog/stateut-wise-distribution-accidental-deaths-natural-causes

### B. Maximum Temperature Data

**Source:** India Meteorological Department (IMD).

Data: Daily maximum temperature GRD files.

Study period: 2001–2025.

Spatial resolution: 1-degree grid.

Season selected: March–May (pre-monsoon/summer).

The daily temperature observations are processed into State-wise seasonal mean maximum temperatures.

### C. State Boundary Data

State boundary shapefile used to assign IMD grid-cell centres to Indian States and Union Territories.

The shapefile is stored locally in `data/boundaries/`.

**Note:** Raw datasets and boundary files are not included in this repository. Users must obtain the source data separately.

## 5. Methodology

The project consists of four major stages.

### Stage 1: Data Preprocessing

- Load and combine mortality datasets.
- Clean and standardise State names.
- Organise mortality records by State and year.
- Calculate cause-specific death rates per lakh population.
- Use 2011 Census population figures as a fixed denominator.

### Stage 2: Clustering Analysis

The following techniques are used:

- StandardScaler – feature standardisation.
- K-Means – State-wise mortality clustering.
- Elbow Method – cluster evaluation using SSE.
- Silhouette Score – cluster separation evaluation.
- Principal Component Analysis (PCA) – dimensionality reduction and visualisation.
- Hierarchical Clustering – dendrogram-based analysis.
- Isolation Forest – potential outlier detection.

Features:
- Lightning death rate per lakh.
- Heat-stroke death rate per lakh.
- Flood death rate per lakh.

### Stage 3: Temperature Analysis

- Process daily IMD maximum temperature GRD files.
- Select March, April and May observations.
- Calculate seasonal mean maximum temperature for each grid cell.
- Assign grid-cell centres to State boundaries.
- Calculate State-wise mean summer maximum temperature.
- Analyse the national State-averaged temperature trend from 2001–2025.

### Stage 4: Regression Analysis

A Linear Regression model is used to investigate the association between summer maximum temperature and heat-stroke mortality.

**Predictor:** Summer mean maximum temperature (°C).

**Target:** Heat-stroke deaths per lakh population.

- Study period: 2001–2014.
- Training period: 2001–2011.
- Testing period: 2012–2014.

Evaluation metrics:
- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R² Score.

The model is also compared with a training-mean baseline.

## 6. Machine Learning Techniques

| Technique | Purpose |
|---|---|
| StandardScaler | Feature scaling |
| K-Means | State clustering |
| PCA | Dimensionality reduction |
| Hierarchical Clustering | Cluster structure analysis |
| Isolation Forest | Outlier detection |
| Linear Regression | Temperature–mortality association |

## 7. Project Structure

```text
disaster_climate_ml_project/
│
├── data/
│   ├── raw/
│   └── boundaries/
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── main.py
│   ├── 01_data_preprocessing.py
│   ├── 02_clustering.py
│   ├── temperature_analysis.py
│   └── 03_regression.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GeoPandas
- SciPy
- Jupyter / VS Code
- Git and GitHub

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/Priyanjana511/india-disaster-mortality-climate-ml.git
```

### Step 2: Navigate to the project folder

```bash
cd india-disaster-mortality-climate-ml
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Add datasets

Place the mortality datasets and IMD GRD files inside:

```text
data/raw/
```

Place the State boundary shapefile and its supporting files inside:

```text
data/boundaries/
```

The boundary file is expected to be named `STATE_BOUNDARY.shp` and contain a `STATE` attribute.

### Step 7: Run the complete pipeline

```bash
python src/main.py
```

The pipeline executes preprocessing, clustering, temperature analysis and regression sequentially.

## 10. Outputs and Visualisations

Generated results are organised into two folders.

**outputs/figures/**

- State-wise mortality heatmap.
- K-Means elbow plot.
- Silhouette score plot.
- PCA cluster visualisation.
- Hierarchical clustering dendrogram.
- Outlier detection plot.
- Temperature trend graph.
- Temperature versus heat-stroke mortality graph.
- Actual versus predicted mortality graph.
- Regression residual plot.

**outputs/tables/**

- Cleaned mortality dataset.
- State population dataset.
- State-year mortality rates.
- Clustering evaluation metrics.
- Final clustering results.
- State-wise temperature dataset.
- Grid-level temperature dataset.
- Regression dataset.
- Regression evaluation metrics.
- Regression test predictions.

## 11. Limitations

- The mortality dataset covers 2001–2014, while temperature data covers 2001–2025. Regression uses only the overlapping years.
- The 2011 Census population is used as a fixed denominator, which is an approximation of annual population.
- Temperature is calculated using equally weighted grid cells assigned to State polygons; it is not population-weighted exposure.
- State boundary changes and historical State-name differences require careful harmonisation.
- Other factors, including humidity, heatwave duration, occupation, healthcare access and reporting practices, may influence mortality.
- Regression identifies statistical association and does not establish causation.
- The short testing period limits the strength of conclusions about future predictive performance.

## 12. Expected Contribution

This project aims to provide a data-driven framework for:

- Identifying differences in disaster mortality across Indian States.
- Grouping States according to their natural-hazard mortality profiles.
- Highlighting potential mortality outliers.
- Exploring the relationship between summer temperature and heat-stroke mortality.
- Supporting State-specific disaster preparedness and early-warning priorities.

## 13. Author

**Priyanjana Paul**

B.Sc. (Hons.) Computer Science  
Ramanujan College, University of Delhi

## 14. Repository

GitHub: https://github.com/Priyanjana511/india-disaster-mortality-climate-ml

---

**Disclaimer:** This project is an academic research and analytical exercise. Its findings should be interpreted in light of the data coverage, methodological assumptions and limitations described above.
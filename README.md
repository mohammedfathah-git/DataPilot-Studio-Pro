# ⚡ DataPilot Studio Pro — Enterprise AI & Autonomous Data Analyst Platform

**DataPilot Studio Pro** is a modern, full-stack Data Analyst platform built with Python, Streamlit, Pandas, Plotly, SciPy, and Scikit-Learn.

Transforming raw, disconnected data into executive-ready business intelligence, statistical inferences, and predictive drivers.

---

## 🌟 Key Architecture & Capabilities

### 1. 📁 Multi-Dataset Workspace & Relational Data Hub
- **Multi-File Simultaneous Ingestion**: Drop multiple CSVs, multi-sheet Excel files, JSON, or Parquet files simultaneously.
- **Relational Table Fusion (Joins & Unions)**: Perform `INNER`, `LEFT`, `RIGHT`, and `FULL OUTER` joins between any two loaded datasets using selected foreign keys.
- **Table Concatenation**: Stack and union datasets row-wise.
- **Curated Enterprise Sample Packs**: Instant 1-click loading for Retail E-commerce (Orders + Customers), Employee HR & Attrition, and Restaurant Tips.

### 2. 🔍 Automated Quality Profiling & Data Health Index
- **0–100 Data Health Score**: Transparent grading across Completeness (35%), Uniqueness (25%), Consistency (20%), and Validity/Outlier hygiene (20%).
- **Semantic Type Classification**: Automatically detects Identifiers, Currency / Monetary dimensions, Percentages, Categorical factors, and Formatted numeric text.
- **Missingness Heatmaps & Distributions**: Visual breakdowns of nullity.
- **Statistical Outlier Scanner**: Tukey's 1.5 IQR anomaly scanner across all numeric attributes.

### 3. ✨ Data Cleaning & Feature Engineering Laboratory
- **1-Click Intelligent Auto-Clean**: Instant deduplication, whitespace trimming, numeric parsing, median/mode imputation, and outlier winsorization.
- **Precision Studio**: Fine-tune individual imputation strategies, casing, and missingness thresholds.
- **Feature Engineering**:
  - Temporal decomposition: extract Year, Month, Quarter, Day of Week, and Weekend flags.
  - Mathematical transforms: Log (log1p), Z-Score standard scaling, Min-Max normalization.
  - Segmentation: Quantile and Equal-Width discretization.
  - Calculated formula engine: evaluate custom Pandas expressions (e.g. `Units_Sold * Unit_Price * (1 - Discount_Pct)`).
- **Lineage & Audit Trail**: Real-time logging of all applied data operations.

### 4. 📊 Statistical Analysis & Hypothesis Testing ("Analyst Brain")
- **Automated Narrative Insights**: Derives top volume concentrations, high-correlation alerts, and extreme skewness.
- **Hypothesis Testing Lab**:
  - Independent Two-Sample T-Test (Welch's) with plain-English business interpretation.
  - One-Way ANOVA across multiple categorical levels.
  - Chi-Square Test of Independence with contingency matrix cross-tabulation.
- **Multivariate Correlation Matrix**: Interactive heatmap with multicollinearity detection.
- **Dynamic Pivot Grid**: Multi-dimensional aggregation with row, column, and metric rollups.

### 5. 📈 Executive Visualization Studio
- Supported architectures: Grouped/Stacked Bar, Multi-Line Trends, Scatter & Bubble with OLS regression trendline, Histograms with KDE/Box marginals, Box Plots, Violin Plots, Donut charts, Treemaps, and Heatmaps.
- **Interactive Global Cross-Filters**: Real-time data slicing across categories.
- **Curated Color Palettes**: *Indigo Modern*, *Cyberpunk Emerald*, *Sunset Warm*, and *Corporate Navy*.
- **Direct Standalone HTML Export**: Download interactive Plotly visualizations for sharing.

### 6. 🤖 Predictive ML & Key Drivers Studio (No-Code AutoML)
- **Automated Target Prediction**: Auto-detects whether the goal is Continuous Regression or Categorical Classification.
- **Model Evaluation**: R², RMSE, MAE for regression; Accuracy, Precision, Recall, F1, and Confusion Matrix for classification.
- **Ranked Key Business Drivers**: Feature importance ranking showing which variables impact business outcomes.
- **Unsupervised K-Means Segmentation**: Group customers or entities into distinct behavioral clusters with centroid profiles.

### 7. 📑 Executive Reports & Reproducibility Suite
- **Stakeholder Executive Brief**: Generate styled HTML executive reports with KPIs, data health breakdown, and key insights.
- **Multi-Sheet Analytical Excel**: Exports cleaned data alongside data dictionary and correlation matrix.
- **Reproducible Python Script**: Exports copy-pasteable Pandas code reproducing the entire session pipeline.

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
python -m pip install streamlit pandas numpy plotly scipy scikit-learn openpyxl
```

### 2. Launch the Application
```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

"""
DataPilot Studio Pro - Automated EDA & Data Quality Profiler
Calculates comprehensive Data Health Score, semantic profiling, missingness patterns,
statistical distributions, and outlier metrics.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats

NUM_CLEAN = r"[,$€£₹%\s]"

def text_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()

def datetime_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

def classify_semantic_type(series: pd.Series) -> str:
    """Infers real-world semantic meaning (ID, Date, Currency, Category, Numeric, Text)."""
    s = series.dropna()
    if s.empty:
        return "Empty"
    
    name = series.name.lower()
    
    # ID detection
    if any(k in name for k in ["id", "uuid", "guid", "code", "key", "number"]) and series.nunique() / max(len(s), 1) > 0.8:
        return "Identifier"
    
    # Datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
    if any(k in name for k in ["date", "time", "timestamp", "year", "month"]):
        return "Date/Time Candidate"
        
    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        if any(k in name for k in ["price", "cost", "salary", "revenue", "sales", "fee", "amount", "budget", "profit"]):
            return "Currency / Monetary"
        if any(k in name for k in ["pct", "percent", "rate", "ratio"]):
            return "Percentage / Rate"
        if series.nunique() <= 10 and series.dtype in [np.int64, np.int32]:
            return "Discrete Numeric (Rating/Level)"
        return "Continuous Numeric"
        
    # Text types
    sample_str = s.astype(str).head(50)
    # Check if currency formatted string
    if sample_str.str.contains(r"^[\$€£₹]", regex=True).mean() > 0.5:
        return "Currency String"
    # Check if numeric-like
    conv = pd.to_numeric(sample_str.str.replace(NUM_CLEAN, "", regex=True), errors="coerce")
    if conv.notna().mean() >= 0.8:
        return "Numeric-Like Text"
        
    # Categorical
    if series.nunique() <= 30 or (series.nunique() / max(len(s), 1) < 0.15):
        return "Categorical"
        
    return "Freeform Text"

def profile_dataset(df: pd.DataFrame) -> dict:
    """Generates an extensive data quality and statistical profile."""
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    
    missing_series = df.isna().sum()
    total_missing = int(missing_series.sum())
    missing_pct = round((total_missing / total_cells) * 100, 2)
    
    duplicates = int(df.duplicated().sum())
    duplicate_pct = round((duplicates / max(rows, 1)) * 100, 2)
    
    # Text checks
    blank_cells = 0
    spacey_cells = 0
    numeric_like = []
    
    for c in text_columns(df):
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        stripped = s.str.strip()
        blank_cells += int((stripped == "").sum())
        spacey_cells += int(((s != stripped) & (stripped != "")).sum())
        conv = pd.to_numeric(s.str.replace(NUM_CLEAN, "", regex=True), errors="coerce")
        if conv.notna().mean() >= 0.85:
            numeric_like.append(c)
            
    # Outlier scan & column stats
    col_profiles = []
    outlier_rows = set()
    num_cols = numeric_columns(df)
    
    for c in df.columns:
        s = df[c]
        non_null_count = int(s.notna().sum())
        null_count = int(s.isna().sum())
        unique_count = int(s.nunique(dropna=True))
        sem_type = classify_semantic_type(s)
        
        info = {
            "Column": c,
            "Raw Type": str(s.dtype),
            "Semantic Type": sem_type,
            "Missing": null_count,
            "Missing (%)": round((null_count / max(rows, 1)) * 100, 2),
            "Unique": unique_count,
            "Cardinality (%)": round((unique_count / max(non_null_count, 1)) * 100, 2),
            "Mean": None,
            "Std": None,
            "Median": None,
            "Min": None,
            "Max": None,
            "Skewness": None,
            "IQR Outliers": 0,
            "Top Value": None,
            "Top Freq (%)": None
        }
        
        if c in num_cols and non_null_count > 1:
            clean_s = s.dropna()
            q1 = float(clean_s.quantile(0.25))
            q3 = float(clean_s.quantile(0.75))
            iqr = q3 - q1
            n_outliers = 0
            if iqr > 0:
                outlier_idx = clean_s[(clean_s < q1 - 1.5 * iqr) | (clean_s > q3 + 1.5 * iqr)].index
                n_outliers = len(outlier_idx)
                outlier_rows.update(outlier_idx)
                
            info.update({
                "Mean": round(float(clean_s.mean()), 3),
                "Std": round(float(clean_s.std()), 3),
                "Median": round(float(clean_s.median()), 3),
                "Min": round(float(clean_s.min()), 3),
                "Max": round(float(clean_s.max()), 3),
                "Skewness": round(float(stats.skew(clean_s)), 3) if len(clean_s) >= 3 else 0.0,
                "IQR Outliers": n_outliers
            })
        elif non_null_count > 0:
            top_val = s.mode(dropna=True)
            if not top_val.empty:
                info["Top Value"] = str(top_val.iloc[0])[:30]
                freq = int((s == top_val.iloc[0]).sum())
                info["Top Freq (%)"] = round((freq / non_null_count) * 100, 1)
                
        col_profiles.append(info)
        
    col_df = pd.DataFrame(col_profiles)
    
    # Data Health Score Computation (0 - 100)
    # 1. Completeness (35 pts)
    completeness_score = max(0, 35 * (1 - (total_missing / total_cells)))
    # 2. Uniqueness (25 pts)
    uniqueness_score = max(0, 25 * (1 - (duplicates / max(rows, 1))))
    # 3. Consistency (20 pts: spaces, blanks, numeric strings)
    consistency_penalties = 0
    if blank_cells > 0: consistency_penalties += min(8, blank_cells / max(rows, 1) * 20)
    if spacey_cells > 0: consistency_penalties += min(6, spacey_cells / max(rows, 1) * 15)
    if len(numeric_like) > 0: consistency_penalties += min(6, len(numeric_like) * 2)
    consistency_score = max(0, 20 - consistency_penalties)
    # 4. Validity & Outlier Hygiene (20 pts)
    outlier_pct = (len(outlier_rows) / max(rows, 1)) * 100
    validity_score = max(0, 20 - min(20, outlier_pct * 1.5))
    
    overall_health = round(completeness_score + uniqueness_score + consistency_score + validity_score, 1)
    
    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    
    # Memory estimation
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = round(mem_bytes / (1024 * 1024), 2)
    
    return {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "missing_cells": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicates,
        "duplicate_pct": duplicate_pct,
        "blank_cells": blank_cells,
        "spacey_cells": spacey_cells,
        "numeric_like": numeric_like,
        "outlier_rows_count": len(outlier_rows),
        "outlier_rows_pct": round((len(outlier_rows) / max(rows, 1)) * 100, 2),
        "constant_columns": constant_cols,
        "memory_mb": mem_mb,
        "health_score": overall_health,
        "health_breakdown": {
            "Completeness (35%)": round(completeness_score, 1),
            "Uniqueness (25%)": round(uniqueness_score, 1),
            "Consistency (20%)": round(consistency_score, 1),
            "Validity (20%)": round(validity_score, 1),
        },
        "columns_summary": col_df
    }

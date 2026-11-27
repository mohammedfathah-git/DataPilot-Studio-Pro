"""
DataPilot Studio Pro - Advanced Statistical Analysis & Hypothesis Testing
Implements T-Tests, ANOVA, Chi-Square Independence, Correlation matrices,
dynamic multidimensional Pivot Tables, and Automated Narrative Insights.
"""
import numpy as np
import pandas as pd
from scipy import stats
from modules.profiler import numeric_columns, text_columns

def run_ttest(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    """Performs two-sample independent T-test across two distinct groups."""
    valid_groups = df[group_col].dropna().unique()
    if len(valid_groups) < 2:
        raise ValueError(f"Column '{group_col}' needs at least 2 distinct groups.")
    
    # Pick first two groups
    g1_val, g2_val = valid_groups[0], valid_groups[1]
    g1 = df[df[group_col] == g1_val][num_col].dropna()
    g2 = df[df[group_col] == g2_val][num_col].dropna()
    
    if len(g1) < 3 or len(g2) < 3:
        raise ValueError("Each group must contain at least 3 non-null records.")
        
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False) # Welch's t-test
    
    is_sig = p_val < 0.05
    diff = g1.mean() - g2.mean()
    pct_diff = (diff / max(abs(g2.mean()), 1e-6)) * 100
    
    if is_sig:
        interp = (
            f"Statistically Significant difference detected (p = {p_val:.4g} < 0.05). "
            f"Group '{g1_val}' (mean {g1.mean():.2f}) is {abs(pct_diff):.1f}% "
            f"{'higher' if diff > 0 else 'lower'} than Group '{g2_val}' (mean {g2.mean():.2f})."
        )
    else:
        interp = (
            f"No statistically significant difference found (p = {p_val:.4g} >= 0.05). "
            f"Any observed difference between '{g1_val}' and '{g2_val}' is likely due to random chance."
        )
        
    return {
        "test": "Welch's Two-Sample T-Test",
        "group_col": group_col,
        "num_col": num_col,
        "g1_name": str(g1_val),
        "g1_count": len(g1),
        "g1_mean": round(float(g1.mean()), 3),
        "g1_std": round(float(g1.std()), 3),
        "g2_name": str(g2_val),
        "g2_count": len(g2),
        "g2_mean": round(float(g2.mean()), 3),
        "g2_std": round(float(g2.std()), 3),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "is_significant": is_sig,
        "interpretation": interp
    }

def run_anova(df: pd.DataFrame, num_col: str, cat_col: str) -> dict:
    """Performs One-Way ANOVA across 3 or more groups."""
    groups = [group.dropna() for _, group in df.groupby(cat_col)[num_col] if len(group.dropna()) >= 3]
    unique_names = [str(k) for k, group in df.groupby(cat_col)[num_col] if len(group.dropna()) >= 3]
    
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 valid categories with >= 3 data points in '{cat_col}'.")
        
    f_stat, p_val = stats.f_oneway(*groups)
    is_sig = p_val < 0.05
    
    if is_sig:
        interp = (
            f"Statistically Significant difference across groups (F = {f_stat:.3f}, p = {p_val:.4g} < 0.05). "
            f"At least one category in '{cat_col}' has a significantly different mean '{num_col}'."
        )
    else:
        interp = (
            f"No significant variance across groups (F = {f_stat:.3f}, p = {p_val:.4g} >= 0.05). "
            f"Group means for '{num_col}' are statistically consistent."
        )
        
    summary_df = df.groupby(cat_col)[num_col].agg(["count", "mean", "std", "min", "max"]).reset_index()
    summary_df.columns = [cat_col, "Sample Count", "Mean", "Std Dev", "Min", "Max"]
    
    return {
        "test": "One-Way ANOVA",
        "cat_col": cat_col,
        "num_col": num_col,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val),
        "is_significant": is_sig,
        "interpretation": interp,
        "group_summary": summary_df
    }

def run_chisquare(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Performs Chi-Square Test of Independence between two categorical features."""
    ct = pd.crosstab(df[col1], df[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Both columns must contain at least 2 distinct categories.")
        
    chi2, p_val, dof, expected = stats.chi2_contingency(ct)
    is_sig = p_val < 0.05
    
    if is_sig:
        interp = (
            f"Significant Association detected (Chi2 = {chi2:.2f}, p = {p_val:.4g} < 0.05, DoF = {dof}). "
            f"The distribution of '{col1}' depends significantly on '{col2}'."
        )
    else:
        interp = (
            f"Independence confirmed (Chi2 = {chi2:.2f}, p = {p_val:.4g} >= 0.05, DoF = {dof}). "
            f"There is no evidence of dependency between '{col1}' and '{col2}'."
        )
        
    return {
        "test": "Chi-Square Test of Independence",
        "col1": col1,
        "col2": col2,
        "chi2": round(float(chi2), 3),
        "p_value": float(p_val),
        "dof": int(dof),
        "is_significant": is_sig,
        "interpretation": interp,
        "contingency_table": ct
    }

def generate_pivot(df: pd.DataFrame, rows: list[str], cols: str | None, values: str, agg: str) -> pd.DataFrame:
    """Generates a dynamic pivot table."""
    if cols:
        pivot = pd.pivot_table(
            df,
            index=rows,
            columns=cols,
            values=values,
            aggfunc=agg,
            fill_value=0,
            margins=True,
            margins_name="Total"
        )
    else:
        pivot = df.groupby(rows)[values].agg(agg).reset_index()
    return pivot

def generate_automated_insights(df: pd.DataFrame) -> list[dict]:
    """Derives key data analyst business observations and anomalies automatically."""
    insights = []
    num_cols = numeric_columns(df)
    cat_cols = text_columns(df)
    
    # 1. Volume & Completeness
    n_rows, n_cols = df.shape
    insights.append({
        "type": "scale",
        "icon": "📐",
        "title": "Dataset Footprint",
        "text": f"The active workspace holds {n_rows:,} records across {n_cols} attributes."
    })
    
    # 2. Dominant Category Check (Concentration Risk)
    for c in cat_cols[:4]:
        val_counts = df[c].value_counts(normalize=True)
        if not val_counts.empty and val_counts.iloc[0] > 0.45:
            top_cat = val_counts.index[0]
            pct = round(val_counts.iloc[0] * 100, 1)
            insights.append({
                "type": "concentration",
                "icon": "🎯",
                "title": f"High Concentration in '{c}'",
                "text": f"'{top_cat}' dominates {pct}% of all records in column '{c}'."
            })
            break
            
    # 3. High Correlation Discovery
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        checked = set()
        strong_pairs = []
        for c1 in num_cols:
            for c2 in num_cols:
                if c1 != c2 and (c2, c1) not in checked:
                    checked.add((c1, c2))
                    val = corr_matrix.loc[c1, c2]
                    if abs(val) >= 0.70 and not np.isnan(val):
                        strong_pairs.append((c1, c2, val))
        for c1, c2, val in strong_pairs[:2]:
            insights.append({
                "type": "correlation",
                "icon": "🔗",
                "title": f"Strong {'Positive' if val > 0 else 'Negative'} Correlation ({val:.2f})",
                "text": f"Strong relationship discovered between '{c1}' and '{c2}' (r = {val:.2f})."
            })
            
    # 4. High Skewness Alert
    for c in num_cols[:4]:
        s = df[c].dropna()
        if len(s) > 10:
            skew_val = stats.skew(s)
            if abs(skew_val) > 2.0:
                insights.append({
                    "type": "distribution",
                    "icon": "⚠️",
                    "title": f"Extreme Skew in '{c}'",
                    "text": f"Column '{c}' is heavily {'right' if skew_val > 0 else 'left'}-skewed (skew = {skew_val:.2f}), suggesting high outliers or power-law dynamics."
                })
                break
                
    return insights

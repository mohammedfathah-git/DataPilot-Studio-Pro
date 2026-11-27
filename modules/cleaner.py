"""
DataPilot Studio Pro - Data Cleaning & Feature Engineering Laboratory
Provides intelligent auto-clean, precision imputation, winsorization, type casting,
datetime extraction, formula generation, and full audit logging.
"""
import re
import numpy as np
import pandas as pd
from modules.profiler import text_columns, numeric_columns

NUM_CLEAN = r"[,$€£₹%\s]"

def clean_dataset(df: pd.DataFrame, options: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies configurable cleaning pipeline and returns (cleaned_dataframe, audit_log).
    """
    out = df.copy()
    log = []

    # 1. Trimming text and normalizing blank strings
    if options.get("trim", False):
        changed = 0
        for c in text_columns(out):
            s = out[c].copy()
            clean_s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            clean_s = clean_s.replace(r"^\s*$", np.nan, regex=True)
            diff = ~(s.isna() & clean_s.isna()) & (s != clean_s)
            changed += int(diff.sum())
            out[c] = clean_s
        log.append(f"Trimmed whitespace & standardized blank cells to nulls ({changed:,} cells altered).")

    # 2. Text Case Normalization
    text_case = options.get("text_case", "None")
    if text_case in ("lowercase", "UPPERCASE", "Title Case"):
        tcols = text_columns(out)
        for c in tcols:
            if text_case == "lowercase":
                out[c] = out[c].map(lambda x: x.lower() if isinstance(x, str) else x)
            elif text_case == "UPPERCASE":
                out[c] = out[c].map(lambda x: x.upper() if isinstance(x, str) else x)
            elif text_case == "Title Case":
                out[c] = out[c].map(lambda x: x.title() if isinstance(x, str) else x)
        log.append(f"Standardized text casing to {text_case} across {len(tcols)} text columns.")

    # 3. Numeric Conversion
    to_numeric = options.get("to_numeric", [])
    conv_success = []
    for c in to_numeric:
        if c in out.columns and not pd.api.types.is_numeric_dtype(out[c]):
            cleaned = out[c].astype(str).str.replace(NUM_CLEAN, "", regex=True)
            out[c] = pd.to_numeric(cleaned, errors="coerce")
            conv_success.append(c)
    if conv_success:
        log.append(f"Converted formatted strings to numbers: {', '.join(conv_success)}.")

    # 4. Datetime Parsing
    to_datetime = options.get("to_datetime", [])
    for c in to_datetime:
        if c in out.columns:
            try:
                out[c] = pd.to_datetime(out[c], errors="coerce")
                log.append(f"Parsed column '{c}' into Datetime objects.")
            except Exception as e:
                log.append(f"Failed to parse datetime for '{c}': {e}")

    # 5. Deduplication
    if options.get("dedupe", False):
        init_rows = len(out)
        dedupe_subset = options.get("dedupe_subset", None)
        out = out.drop_duplicates(subset=dedupe_subset)
        dropped = init_rows - len(out)
        log.append(f"Removed {dropped:,} duplicate rows.")

    # 6. Drop constant columns
    if options.get("drop_constant", False):
        constants = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
        if constants:
            out = out.drop(columns=constants)
            log.append(f"Dropped {len(constants)} zero-variance column(s): {', '.join(constants)}.")

    # 7. Column missing threshold
    col_thr = options.get("col_thr", 100)
    if col_thr < 100:
        high_missing = [c for c in out.columns if (out[c].isna().mean() * 100) > col_thr]
        if high_missing:
            out = out.drop(columns=high_missing)
            log.append(f"Dropped {len(high_missing)} column(s) exceeding {col_thr}% missing values: {', '.join(high_missing)}.")

    # 8. Numeric Imputation
    num_cols = [c for c in numeric_columns(out) if out[c].isna().any()]
    ns = options.get("num_strategy", "Leave as is")
    drop_num_subset = []
    
    if num_cols and ns != "Leave as is":
        if ns == "Drop rows with missing numbers":
            drop_num_subset = num_cols
        else:
            for c in num_cols:
                if ns == "Fill with median":
                    val = out[c].median()
                elif ns == "Fill with mean":
                    val = out[c].mean()
                elif ns == "Fill with 0":
                    val = 0
                elif ns == "Forward fill":
                    out[c] = out[c].ffill()
                    continue
                elif ns == "Linear Interpolation":
                    out[c] = out[c].interpolate(method="linear")
                    continue
                else:
                    val = 0
                out[c] = out[c].fillna(val)
            log.append(f"Numeric imputation: applied '{ns}' across {len(num_cols)} column(s).")

    # 9. Categorical Imputation
    cat_cols = [c for c in text_columns(out) if out[c].isna().any()]
    ts = options.get("txt_strategy", "Leave as is")
    drop_txt_subset = []
    
    if cat_cols and ts != "Leave as is":
        if ts == "Drop rows with missing text":
            drop_txt_subset = cat_cols
        else:
            for c in cat_cols:
                if ts == "Fill with mode (most frequent)":
                    mode_val = out[c].mode(dropna=True)
                    val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                elif ts == "Fill with 'Unknown'":
                    val = "Unknown"
                elif ts == "Fill with 'Missing'":
                    val = "Missing"
                elif ts == "Forward fill":
                    out[c] = out[c].ffill()
                    continue
                else:
                    val = "Unknown"
                out[c] = out[c].fillna(val)
            log.append(f"Categorical imputation: applied '{ts}' across {len(cat_cols)} column(s).")

    # Drop subsets if requested
    drop_all_missing = list(set(drop_num_subset + drop_txt_subset))
    if drop_all_missing:
        init_rows = len(out)
        out = out.dropna(subset=drop_all_missing)
        log.append(f"Dropped {init_rows - len(out):,} rows with missing values in specified columns.")

    # 10. Outlier Treatment (Remove or Winsorize / Cap)
    outlier_action = options.get("outlier_action", "Leave as is")
    if outlier_action != "Leave as is":
        num_targets = numeric_columns(out)
        if outlier_action == "Drop outlier rows (IQR rule)":
            mask = pd.Series(False, index=out.index)
            for c in num_targets:
                s = out[c].dropna()
                if len(s) < 5: continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0: continue
                mask |= (out[c] < q1 - 1.5 * iqr) | (out[c] > q3 + 1.5 * iqr)
            n_dropped = int(mask.sum())
            out = out[~mask]
            log.append(f"Removed {n_dropped:,} rows identified as IQR outliers.")
        elif outlier_action == "Winsorize (Cap at 1st & 99th percentiles)":
            capped_cols = 0
            for c in num_targets:
                s = out[c].dropna()
                if len(s) < 10: continue
                p01, p99 = s.quantile(0.01), s.quantile(0.99)
                out[c] = out[c].clip(lower=p01, upper=p99)
                capped_cols += 1
            log.append(f"Winsorized/capped extreme outliers at 1st & 99th percentiles across {capped_cols} numeric column(s).")

    if not log:
        log.append("No cleaning rules applied; dataset remains unchanged.")

    return out.reset_index(drop=True), log

def engineer_features(df: pd.DataFrame, action: str, params: dict) -> tuple[pd.DataFrame, str]:
    """
    Applies data analyst feature transformations (dates, math, binning, formulas).
    """
    out = df.copy()
    msg = ""

    if action == "date_decompose":
        col = params.get("column")
        if col in out.columns:
            s = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}_year"] = s.dt.year
            out[f"{col}_month"] = s.dt.month
            out[f"{col}_quarter"] = s.dt.quarter
            out[f"{col}_day_name"] = s.dt.day_name()
            out[f"{col}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype(int)
            msg = f"Extracted 5 temporal features from '{col}' (year, month, quarter, day_name, is_weekend)."

    elif action == "math_transform":
        col = params.get("column")
        func = params.get("func", "Log (log1p)")
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col]
            if func == "Log (log1p)":
                out[f"{col}_log"] = np.log1p(np.maximum(s, 0))
                msg = f"Created '{col}_log' via natural log transform."
            elif func == "Standard Scaling (Z-score)":
                std = s.std()
                if std > 0:
                    out[f"{col}_zscore"] = ((s - s.mean()) / std).round(4)
                    msg = f"Created standard normal Z-score column '{col}_zscore'."
            elif func == "Min-Max Normalization (0-1)":
                rng = s.max() - s.min()
                if rng > 0:
                    out[f"{col}_norm"] = ((s - s.min()) / rng).round(4)
                    msg = f"Created min-max normalized column '{col}_norm'."

    elif action == "binning":
        col = params.get("column")
        n_bins = params.get("bins", 4)
        method = params.get("method", "Equal Width")
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].dropna()
            if method == "Equal Width":
                out[f"{col}_binned"] = pd.cut(out[col], bins=n_bins, labels=[f"Tier_{i+1}" for i in range(n_bins)])
            else:
                out[f"{col}_binned"] = pd.qcut(out[col], q=n_bins, labels=[f"Q_{i+1}" for i in range(n_bins)], duplicates="drop")
            msg = f"Discretized '{col}' into {n_bins} segments using {method}."

    elif action == "calculated_column":
        new_col = params.get("new_col_name", "calculated_metric")
        expr = params.get("expression", "")
        try:
            # Safe evaluation using pandas eval
            out[new_col] = out.eval(expr)
            msg = f"Computed calculated column '{new_col}' from expression: `{expr}`."
        except Exception as e:
            raise ValueError(f"Expression evaluation error: {e}")

    return out, msg

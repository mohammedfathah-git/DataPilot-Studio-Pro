"""
DataPilot Studio Pro - Multi-dataset Manager
Handles ingestion, multi-file storage, relational joins, concatenation, and sample datasets.
"""
import io
import re
import numpy as np
import pandas as pd
import plotly.express as px

NUM_CLEAN = r"[,$€£₹%\s]"

def load_file(data: bytes, filename: str, sheet=None) -> dict:
    """
    Parses an uploaded file. If Excel has multiple sheets, returns a dict of sheet_name -> DataFrame.
    For CSV/JSON/Parquet, returns {filename: DataFrame}.
    """
    lower = filename.lower()
    results = {}
    
    if lower.endswith(".csv"):
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        if df is None:
            raise ValueError(f"Could not parse CSV: {filename}")
        results[clean_name(filename)] = df
        
    elif lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(data))
        if sheet is not None:
            results[clean_name(f"{filename}_{sheet}")] = pd.read_excel(xl, sheet_name=sheet)
        else:
            for sname in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sname)
                t_name = f"{filename}_{sname}" if len(xl.sheet_names) > 1 else filename
                results[clean_name(t_name)] = df
                
    elif lower.endswith(".json"):
        df = pd.read_json(io.BytesIO(data))
        results[clean_name(filename)] = df
        
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(data))
        results[clean_name(filename)] = df
        
    else:
        raise ValueError(f"Unsupported format: {filename}")
        
    return results

def clean_name(name: str) -> str:
    """Sanitize dataset name for clean tab and variable reference."""
    clean = re.sub(r"[^\w\.-]", "_", name)
    clean = re.sub(r"\.(csv|xlsx|xls|json|parquet)$", "", clean, flags=re.I)
    return clean

def get_sample_datasets() -> dict:
    """Provides rich, curated sample datasets covering diverse analytical scenarios."""
    datasets = {}
    
    # 1. Retail Sales & Customers
    np.random.seed(42)
    n_orders = 350
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    categories = ["Electronics", "Office Supplies", "Furniture", "Apparel"]
    regions = ["North", "South", "East", "West"]
    
    orders_df = pd.DataFrame({
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n_orders)],
        "Customer_ID": [f"CUST-{np.random.randint(101, 150)}" for _ in range(n_orders)],
        "Order_Date": np.random.choice(dates, size=n_orders),
        "Category": np.random.choice(categories, size=n_orders, p=[0.35, 0.25, 0.20, 0.20]),
        "Region": np.random.choice(regions, size=n_orders),
        "Units_Sold": np.random.randint(1, 15, size=n_orders),
        "Unit_Price": np.random.choice([15.99, 45.50, 120.00, 299.99, 49.00], size=n_orders),
        "Discount_Pct": np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n_orders),
        "Payment_Method": np.random.choice(["Credit Card", "PayPal", "Bank Transfer"], size=n_orders),
    })
    orders_df["Total_Sales"] = (orders_df["Units_Sold"] * orders_df["Unit_Price"] * (1 - orders_df["Discount_Pct"])).round(2)
    # Add minor realistic noise (a few missing or duplicate values)
    orders_df.loc[12, "Total_Sales"] = np.nan
    orders_df.loc[45, "Discount_Pct"] = np.nan
    datasets["ecommerce_orders"] = orders_df

    customers_df = pd.DataFrame({
        "Customer_ID": [f"CUST-{i}" for i in range(101, 150)],
        "Segment": np.random.choice(["Consumer", "Corporate", "Small Business"], size=49),
        "Customer_Age": np.random.randint(22, 68, size=49),
        "Satisfaction_Score": np.random.choice([1, 2, 3, 4, 5], size=49, p=[0.05, 0.1, 0.25, 0.4, 0.2]),
        "Membership": np.random.choice(["Gold", "Silver", "Bronze", "None"], size=49),
    })
    datasets["ecommerce_customers"] = customers_df

    # 2. Tips Dataset
    datasets["restaurant_tips"] = px.data.tips()

    # 3. Employee HR Analytics
    n_emp = 200
    depts = ["Engineering", "Sales", "Marketing", "HR", "Product"]
    education = ["Bachelor", "Master", "PhD", "High School"]
    hr_df = pd.DataFrame({
        "Employee_ID": [f"EMP-{2000 + i}" for i in range(n_emp)],
        "Department": np.random.choice(depts, size=n_emp),
        "Education": np.random.choice(education, size=n_emp, p=[0.5, 0.35, 0.05, 0.1]),
        "Years_Experience": np.random.randint(1, 25, size=n_emp),
        "Salary": np.random.randint(45000, 160000, size=n_emp),
        "Performance_Rating": np.random.choice([1, 2, 3, 4, 5], size=n_emp, p=[0.05, 0.15, 0.50, 0.20, 0.10]),
        "Attrition": np.random.choice(["Yes", "No"], size=n_emp, p=[0.18, 0.82]),
        "Monthly_Hours": np.zeros(n_emp, dtype=int)
    })
    
# Generate realistic salary based on experience
    hr_df["Salary"] = (
        40000
        + hr_df["Years_Experience"] * 4500
        + np.random.normal(0, 8000, n_emp)
    ).astype(int)

    # Generate meaningful monthly working hours
    department_workload = {
        "Engineering": 15,
        "Sales": 22,
        "Marketing": 8,
        "HR": -5,
        "Product": 13
    }

    hr_df["Monthly_Hours"] = (
        165
        + hr_df["Department"].map(department_workload)
        + (hr_df["Performance_Rating"] - 3) * 3
        + hr_df["Years_Experience"] * 0.3
        + np.random.normal(0, 7, n_emp)
    ).round().clip(120, 260).astype(int)

    datasets["employee_hr"] = hr_df


    return datasets

def join_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner",
    suffixes: tuple = ("_x", "_y")
) -> pd.DataFrame:
    """Performs relational merge between two datasets."""
    return pd.merge(
        left_df,
        right_df,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=suffixes
    )

def concat_datasets(dfs: list[pd.DataFrame], ignore_index: bool = True) -> pd.DataFrame:
    """Concatenates multiple datasets row-wise (union)."""
    return pd.concat(dfs, axis=0, ignore_index=ignore_index)

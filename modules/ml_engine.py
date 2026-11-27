"""
DataPilot Studio Pro - No-Code Machine Learning & Predictive Insights Engine
Implements automated Regression, Classification, Feature Importance analysis,
What-If prediction simulator, and K-Means segmentation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from modules.profiler import numeric_columns


def prepare_ml_data(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """
    Validates, cleans, encodes and prepares data for machine learning.
    """

    # 1. Validate target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if not feature_cols:
        raise ValueError("Please select at least one predictor feature.")

    # 2. Remove duplicate feature selections
    feature_cols = list(dict.fromkeys(feature_cols))

    # 3. Validate feature columns
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    if target_col in feature_cols:
        raise ValueError("Target column cannot also be a predictor.")

    # 4. Select required columns
    data = df[feature_cols + [target_col]].copy()

    # 5. Handle infinite values
    data = data.replace([np.inf, -np.inf], np.nan)

    # 6. Remove rows with missing values
    data = data.dropna()

    if len(data) < 15:
        raise ValueError(
            "Need at least 15 complete rows to train a model."
        )

    # 7. Separate target and predictors
    y = data[target_col]
    X_raw = data[feature_cols].copy()

    if y.nunique() < 2:
        raise ValueError(
            "Target must contain at least two unique values."
        )

    # 8. Convert datetime features into useful components
    for col in X_raw.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns:

        X_raw[f"{col}_year"] = X_raw[col].dt.year
        X_raw[f"{col}_month"] = X_raw[col].dt.month
        X_raw[f"{col}_day"] = X_raw[col].dt.day

        X_raw.drop(columns=[col], inplace=True)

    # 9. Encode categorical features
    X = pd.get_dummies(
        X_raw,
        drop_first=True,
        dtype=int
    )

    if X.empty:
        raise ValueError(
            "No usable predictor features were found."
        )

    # 10. Detect ML task
    is_regression = (
        pd.api.types.is_numeric_dtype(y)
        and y.nunique() > 10
    )

    return X, y, is_regression, X_raw



def train_predictor(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str]
) -> dict:
    """
    Trains and compares multiple ML models.
    Automatically selects the best-performing model.
    """

    # 1. Prepare data
    X, y, is_regression, X_raw = prepare_ml_data(
        df, target_col, feature_cols
    )

    # 2. Split data
    if not is_regression:
        class_counts = y.value_counts()
        n_classes = y.nunique()
        n_test = int(np.ceil(len(y) * 0.25))
        n_train = len(y) - n_test

        can_stratify = (
            class_counts.min() >= 2
            and n_test >= n_classes
            and n_train >= n_classes
        )
    else:
        can_stratify = False

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None
    )

    results = {}
    trained_models = {}

    # 3. Regression models
    if is_regression:

        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )
            mae = mean_absolute_error(y_test, predictions)

            results[name] = {
                "R² Score": round(float(r2), 4),
                "RMSE": round(float(rmse), 4),
                "MAE": round(float(mae), 4)
            }

            trained_models[name] = model

        # Select model with highest R²
        best_model_name = max(
            results,
            key=lambda name: results[name]["R² Score"]
        )

    # 4. Classification models
    else:

        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        }

        for name, model in models.items():

            # Scale data for Logistic Regression
            if name == "Logistic Regression":
                scaler = StandardScaler()

                X_train_model = scaler.fit_transform(X_train)
                X_test_model = scaler.transform(X_test)
            else:
                scaler = None
                X_train_model = X_train
                X_test_model = X_test

            model.fit(X_train_model, y_train)
            predictions = model.predict(X_test_model)

            acc = accuracy_score(y_test, predictions)
            prec = precision_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
            rec = recall_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
            f1 = f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )

            results[name] = {
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1-Score": round(float(f1), 4)
            }

            trained_models[name] = {
                "model": model,
                "scaler": scaler
            }

        # Select model with highest weighted F1
        best_model_name = max(
            results,
            key=lambda name: results[name]["F1-Score"]
        )

    # 5. Retrieve selected model
    if is_regression:
        best_model = trained_models[best_model_name]
        importance_model = best_model
    else:
        best_model = trained_models[best_model_name]["model"]
        importance_model = best_model

    
    # 6. Calculate feature importance using permutation importance
    from sklearn.inspection import permutation_importance

    if is_regression:
        importance_result = permutation_importance(
            best_model,
            X_test,
            y_test,
            scoring="r2",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        X_test_for_importance = (
            trained_models[best_model_name]["scaler"].transform(X_test)
            if trained_models[best_model_name]["scaler"] is not None
            else X_test
        )

        importance_result = permutation_importance(
            best_model,
            X_test_for_importance,
            y_test,
            scoring="f1_weighted",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

    fi_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_result.importances_mean
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)


    # 7. Classification confusion matrix
    cm = None

    if not is_regression:

        labels = sorted(
            set(y_train) | set(y_test)
        )

        predictions = best_model.predict(
            trained_models[best_model_name]["scaler"].transform(X_test)
            if trained_models[best_model_name]["scaler"] is not None
            else X_test
        )

        cm = pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=labels
            ),
            index=[f"Actual {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels]
        )

    # 8. Return results
    return {
        "is_regression": is_regression,
        "target_col": target_col,
        "metrics": results[best_model_name],
        "model_comparison": results,
        "best_model_name": best_model_name,
        "feature_importance": fi_df,
        "confusion_matrix": cm,
        "model": best_model,
        "scaler": (
            trained_models[best_model_name]["scaler"]
            if not is_regression else None
        ),
        "feature_columns": list(X.columns),
        "raw_features": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

def run_clustering(df: pd.DataFrame, feature_cols: list[str], k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies K-Means clustering and returns segmented dataset and cluster profile matrix."""
    data = df[feature_cols].dropna().copy()
    if len(data) < k * 3:
        raise ValueError(f"Need at least {k * 3} complete rows to segment into {k} clusters.")
        
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    
    data["Cluster"] = [f"Cluster {c+1}" for c in clusters]
    
    # Cluster profiles (mean values per cluster)
    profiles = data.groupby("Cluster").mean()
    profiles["Count"] = data["Cluster"].value_counts()
    profiles["Share (%)"] = (profiles["Count"] / len(data) * 100).round(1)
    
    return data, profiles.reset_index()

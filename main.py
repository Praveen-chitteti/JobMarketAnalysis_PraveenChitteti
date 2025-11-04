
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so figures can be saved in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


# Optional NLTK for simple text cleaning/token length
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


warnings.filterwarnings("ignore")


def ensure_directories(base_dir: str) -> Dict[str, str]:
    """Ensure required directories exist and return their paths."""
    paths = {
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
        "notebooks": os.path.join(base_dir, "notebooks"),
        "tableau": os.path.join(base_dir, "tableau_exports"),
        "figures": os.path.join(base_dir, "figures"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists, else return None."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read CSV at {path}: {e}")
            return None
    return None


def parse_salary(value) -> Optional[float]:
    """Parse various salary text formats to annual numeric salary.

    Heuristics supported:
    - Ranges like "$50k-$60k", "$70,000 - $90,000"
    - Hourly rates like "$30/hr", "$45 per hour" (converted using 2080 hours/year)
    - Monthly "$5,000/mo" (x12)
    - Daily "$300/day" (x260 workdays/year)
    - Plain numbers or with currency symbols/commas
    Returns float or np.nan if unparseable.
    """
    if pd.isna(value):
        return np.nan
    try:
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        if not s:
            return np.nan

        # Detect units
        is_hourly = any(u in s for u in ["/hr", "per hour", "hourly", " hr", "hour"])
        is_monthly = any(u in s for u in ["/mo", "per month", "monthly", " month"])
        is_daily = any(u in s for u in ["/day", "per day", "daily"]) 

        # Replace common tokens
        s = s.replace(",", "")
        s = s.replace("$", "")
        s = s.replace("usd", "")
        s = s.replace("k", "000")

        # Extract numbers
        nums = [float(n) for n in re.findall(r"\d+\.?\d*", s)]
        if not nums:
            return np.nan
        # Average range if two provided
        amount = np.mean(nums) if len(nums) >= 2 else nums[0]

        # Convert based on unit
        if is_hourly:
            amount *= 2080.0
        elif is_monthly:
            amount *= 12.0
        elif is_daily:
            amount *= 260.0
        # Otherwise assume annual
        return float(amount)
    except Exception:
        return np.nan


def clean_text(text: Optional[str]) -> str:
    """Basic text cleanup for job descriptions: strip HTML, lower, remove non-letters.
    If NLTK is available, remove stopwords and tokenize. Returns cleaned string."""
    if pd.isna(text):
        return ""
    s = str(text)
    # Remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Lowercase and keep letters/spaces
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if _NLTK_AVAILABLE:
        try:
            # Try to ensure resources are ready
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            tokens = word_tokenize(s)
            stops = set(stopwords.words("english"))
            tokens = [t for t in tokens if t not in stops]
            return " ".join(tokens)
        except Exception:
            return s
    return s


def load_and_clean_data(paths: Dict[str, str]) -> pd.DataFrame:
    """Load cleaned data if available, else load raw and perform cleaning, saving cleaned CSV."""
    raw_path = os.path.join(paths["data"], "job_listings.csv")
    clean_path = os.path.join(paths["data"], "job_listings_clean.csv")

    # Prefer already-cleaned file
    df = safe_read_csv(clean_path)
    if df is not None:
        print(f"Loaded cleaned dataset: {clean_path} (rows={len(df)})")
        return df

    # Fallback to raw data
    df = safe_read_csv(raw_path)
    if df is None:
        raise FileNotFoundError(
            f"Dataset not found. Please place 'job_listings.csv' in {paths['data']} or provide 'job_listings_clean.csv'."
        )

    print(f"Loaded raw dataset: {raw_path} (rows={len(df)})")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Dropped duplicates: {before - len(df)}")

    # Standardize column names for consistency
    df.columns = [c.strip() for c in df.columns]

    # Identify expected columns if present
    expected_cols = [
        "jobType", "location", "salary", "rating", "company", "jobDescription",
    ]

    # Parse salary to numeric
    if "salary" in df.columns:
        df["salary"] = df["salary"].apply(parse_salary)

    # Clean job descriptions and add simple feature: description length
    if "jobDescription" in df.columns:
        df["jobDescription_clean"] = df["jobDescription"].apply(clean_text)
        df["desc_length"] = df["jobDescription_clean"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    else:
        df["desc_length"] = 0

    # Handle missing values
    # Numeric: fill with mean; Categorical: fill with mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
        else:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    # Save cleaned data
    df.to_csv(clean_path, index=False)
    print(f"Saved cleaned dataset to: {clean_path}")
    return df


def encode_categoricals(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode specific categorical columns if present, return DataFrame and encoders."""
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for c in cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c + "_le"] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            print(f"Warning: Column '{c}' not found; skipping label encoding for it.")
    return df, encoders


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Split the dataset into train/test for the specified target and features."""
    # Keep only available feature columns
    features_available = [c for c in feature_cols if c in df.columns]
    if not features_available:
        raise ValueError("No valid feature columns available for modeling.")
    X = df[features_available].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return X_train, X_test, y_train, y_test, features_available


def train_jobtype_model(
    df: pd.DataFrame,
    models_dir: str,
    figures_dir: str,
) -> Optional[Dict[str, float]]:
    """Train a Logistic Regression model to classify job type. Save model and confusion matrix plot.
    Returns classification metrics or None if not possible.
    """
    if "jobType" not in df.columns:
        print("jobType column not found; skipping classification model.")
        return None

    # Encode target
    y_le = LabelEncoder()
    df["jobType_target"] = y_le.fit_transform(df["jobType"].astype(str))

    # Encode select categoricals for features
    df_enc, _ = encode_categoricals(df, ["location", "company"])  # features only

    # Select features (use numeric, encoded categoricals, and simple text len)
    candidate_features = [
        "rating", "salary", "desc_length", "location_le", "company_le"
    ]
    # Some datasets may not have rating/salary; handled in cleaning
    try:
        X_train, X_test, y_train, y_test, feature_names = split_dataset(
            df_enc, target_col="jobType_target", feature_cols=candidate_features
        )
    except ValueError as e:
        print(f"Skipping classification model due to feature issue: {e}")
        return None

    # Train Logistic Regression
    print("Training Logistic Regression for job type classification...")
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = y_le.classes_
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Job Type Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(figures_dir, "confusion_matrix_jobtype.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # Save model
    model_path = os.path.join(models_dir, "model_jobtype.pkl")
    dump({"model": clf, "feature_names": feature_names, "label_encoder": y_le}, model_path)
    print(f"Saved job type model to: {model_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def get_xgb_regressor():
    """Return an XGBoost regressor instance or None if import fails."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None


def train_salary_model(
    df: pd.DataFrame,
    models_dir: str,
    figures_dir: str,
) -> Optional[Dict[str, float]]:
    """Train a regression model to predict salary. Prefer XGBoost, fallback to RandomForest.
    Saves model and feature importance plot. Returns metrics or None if not possible.
    """
    if "salary" not in df.columns:
        print("salary column not found; skipping salary regression model.")
        return None

    # Encode select categoricals for features (jobType, location, company)
    df_enc, _ = encode_categoricals(df, ["jobType", "location", "company"])  # use encoded forms

    candidate_features = [
        "rating", "desc_length", "jobType_le", "location_le", "company_le"
    ]

    try:
        X_train, X_test, y_train, y_test, feature_names = split_dataset(
            df_enc, target_col="salary", feature_cols=candidate_features
        )
    except ValueError as e:
        print(f"Skipping regression model due to feature issue: {e}")
        return None

    # Try XGBoost first
    reg = get_xgb_regressor()
    model_name = "XGBoostRegressor" if reg is not None else "RandomForestRegressor"
    if reg is None:
        reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

    print(f"Training {model_name} for salary prediction...")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    from math import sqrt
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Feature importance (if available)
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, x="importance", y="feature", color="#4472C4")
        plt.title("Feature Importance - Salary Regression")
        plt.tight_layout()
        fi_path = os.path.join(figures_dir, "feature_importance_salary.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"Saved feature importance plot to: {fi_path}")

    # Save model
    model_path = os.path.join(models_dir, "model_salary.pkl")
    dump({"model": reg, "feature_names": feature_names}, model_path)
    print(f"Saved salary model to: {model_path}")

    return {"rmse": float(rmse), "r2": float(r2)}


def create_tableau_export(df: pd.DataFrame, tableau_dir: str) -> str:
    """Create summary export for Tableau: location, jobType, company, avg_salary, avg_rating."""
    cols_needed = ["location", "jobType", "company", "salary", "rating"]
    present = [c for c in cols_needed if c in df.columns]
    if not set(["location", "salary", "rating"]).issubset(df.columns):
        # Cannot construct meaningful export without these
        simple = df[[c for c in df.columns if c in ["location", "salary", "rating"]]].copy()
        simple.rename(columns={"salary": "avg_salary", "rating": "avg_rating"}, inplace=True)
        out_path = os.path.join(tableau_dir, "job_insights.csv")
        simple.to_csv(out_path, index=False)
        print("Tableau export lacked some columns; saved minimal export.")
        print("Tableau export file created successfully!")
        return out_path

    summary = (
        df.groupby([c for c in ["location", "jobType", "company"] if c in df.columns])
        .agg(avg_salary=("salary", "mean"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    out_path = os.path.join(tableau_dir, "job_insights.csv")
    summary.to_csv(out_path, index=False)
    print("Tableau export file created successfully!")
    return out_path


def print_actionable_insights(df: pd.DataFrame, cls_metrics: Optional[Dict[str, float]], reg_metrics: Optional[Dict[str, float]]) -> None:
    """Print 3–5 actionable insights derived from the data and model results."""
    print("\nActionable Insights:")

    # Top-paying locations
    if {"location", "salary"}.issubset(df.columns):
        top_locs = (
            df.groupby("location")["salary"].mean().sort_values(ascending=False).head(3).index.tolist()
        )
        if top_locs:
            print(f"- Top-paying locations: {', '.join(top_locs)}")

    # High-demand roles (by frequency)
    if "jobType" in df.columns:
        top_roles = df["jobType"].value_counts().head(3).index.tolist()
        if top_roles:
            print(f"- High-demand roles: {', '.join(top_roles)}")

    # Companies with high rating and pay
    if {"company", "salary", "rating"}.issubset(df.columns):
        comp_stats = df.groupby("company").agg(avg_sal=("salary", "mean"), avg_rat=("rating", "mean"))
        med_sal = comp_stats["avg_sal"].median()
        med_rat = comp_stats["avg_rat"].median()
        winners = comp_stats[(comp_stats["avg_sal"] >= med_sal) & (comp_stats["avg_rat"] >= med_rat)].sort_values(
            ["avg_sal", "avg_rat"], ascending=False
        ).head(3).index.tolist()
        if winners:
            print(f"- Companies with high rating and pay correlation: {', '.join(winners)}")

    # Model quality notes
    if cls_metrics is not None:
        print(
            f"- Classification model performance: Acc={cls_metrics['accuracy']:.2f}, "
            f"Prec={cls_metrics['precision']:.2f}, Rec={cls_metrics['recall']:.2f}, F1={cls_metrics['f1']:.2f}"
        )
    if reg_metrics is not None:
        print(f"- Regression model performance: RMSE={reg_metrics['rmse']:.0f}, R²={reg_metrics['r2']:.2f}")

    # Text insight: average description length by role (if available)
    if {"jobType", "desc_length"}.issubset(df.columns):
        long_desc_roles = (
            df.groupby("jobType")["desc_length"].mean().sort_values(ascending=False).head(2).index.tolist()
        )
        if long_desc_roles:
            print(f"- Roles with longest job descriptions: {', '.join(long_desc_roles)}")


def main():
    # Resolve base directory and ensure structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = ensure_directories(base_dir)

    print("\n=== Job Market Analysis: Start ===")
    # Load and clean data
    df = load_and_clean_data(paths)
    print(f"Data shape after cleaning: {df.shape}")

    # Train models
    cls_metrics = train_jobtype_model(df, models_dir=paths["models"], figures_dir=paths["figures"])
    reg_metrics = train_salary_model(df, models_dir=paths["models"], figures_dir=paths["figures"])

    # Tableau export
    create_tableau_export(df, tableau_dir=paths["tableau"]) 

    # Insights
    print_actionable_insights(df, cls_metrics, reg_metrics)

    print("=== Job Market Analysis: Done ===\n")


if __name__ == "__main__":
    main()

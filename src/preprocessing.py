"""Reusable Census-Income preprocessing for controlled comparisons."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "age", "wage_per_hour", "capital_gains", "capital_losses",
    "dividends_from_stocks", "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
MONEY_COLUMNS = [
    "wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks"
]
MIGRATION_COLUMNS = [
    "migration_code_change_in_msa", "migration_code_change_in_reg",
    "migration_code_move_within_reg", "migration_prev_res_in_sunbelt",
]
NUMERIC_CODE_CATEGORIES = [
    "detailed_industry_recode", "detailed_occupation_recode",
    "own_business_or_self_employed", "veterans_benefits", "year",
]
CONSTANT_COLUMNS = ["family_members_under_18"]
# Informative in only one survey year: the other year has a single value for
# every row (1995: "Not in universe under 1 year old"; 1994: "Children or
# Armed Forces"). Dropped so all features mean the same thing in both years.
YEAR_SPECIFIC_COLUMNS = [
    "live_in_this_house_1_year_ago", "full_or_part_time_employment_stat",
]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert placeholders and numeric category codes; keep original rows."""
    X = df.drop(columns="income_class").copy()
    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].str.strip().replace("?", np.nan)
    for col in NUMERIC_CODE_CATEGORIES:
        X[col] = X[col].astype(str)
    labels = df["income_class"].str.strip()
    if labels.isna().any() or not set(labels.unique()).issubset({"- 50000.", "50000+."}):
        raise ValueError("Unexpected income class label")
    y = labels.eq("50000+.").astype(int)
    return X, y


def select_features(X: pd.DataFrame, include_migration: bool) -> pd.DataFrame:
    """Drop the adult-only constant and year-specific columns, then optionally
    drop migration columns."""
    columns_to_drop = CONSTANT_COLUMNS + YEAR_SPECIFIC_COLUMNS
    if not include_migration:
        columns_to_drop += MIGRATION_COLUMNS
    return X.drop(columns=columns_to_drop)


def build_preprocessor(X: pd.DataFrame, log_money: bool) -> ColumnTransformer:
    """Fit imputation, scaling and one-hot encoding on training rows only."""
    money_columns = [col for col in MONEY_COLUMNS if col in X.columns]
    other_numeric = [
        col for col in NUMERIC_COLUMNS if col not in MONEY_COLUMNS and col in X.columns
    ]
    categorical = [col for col in X.columns if col not in NUMERIC_COLUMNS]

    money_steps = [("impute", SimpleImputer(strategy="median"))]
    if log_money:
        money_steps.append(
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        )
    money_steps.append(("scale", StandardScaler()))
    transformers = []
    if money_columns:
        transformers.append(("money", Pipeline(money_steps), money_columns))
    if other_numeric:
        transformers.append(("other_numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), other_numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

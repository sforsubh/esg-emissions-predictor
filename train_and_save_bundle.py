import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# ---------- 1) Load data ----------
data_path = Path("data") / "emissions_high_granularity.csv"
df = pd.read_csv(data_path)

drop_cols = [
    "lei","source","parent_entity","reporting_entity","year",
    "product_emissions_MtCO2","flaring_emissions_MtCO2","venting_emissions_MtCO2",
    "own_fuel_use_emissions_MtCO2","fugitive_methane_emissions_MtCO2e",
    "fugitive_methane_emissions_MtCH4","total_operational_emissions_MtCO2e","emission_intensity"
]
df = df.drop(columns=drop_cols, errors="ignore")

# Winsorize production_value
lo, hi = df["production_value"].quantile([0.01, 0.99])
df["production_value"] = df["production_value"].clip(lo, hi)

# Rare category grouping
for cat in ["commodity","parent_type","production_unit"]:
    freqs = df[cat].value_counts(normalize=True)
    df[cat] = df[cat].replace(freqs[freqs < 0.01].index, "Other")

# ---------- 2) Features & target ----------
X = df.drop(columns=["total_emissions_MtCO2e"], errors="ignore")
y = df["total_emissions_MtCO2e"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3) Preprocessor ----------
num_feats = X.select_dtypes(include="number").columns.tolist()
cat_feats = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_feats)
])

# ---------- 4) Models with log1p-y ----------
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)

rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1))
])

xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        reg_lambda=1,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ))
])

# Trigger rebuild after Python version change

import sys, subprocess, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ---------------- Page setup & minimal CSS ----------------
st.set_page_config(
    page_title="ESG Emissions Predictor",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
DEBUG = False  # flip to True only when you want on-page debug messages


# ---------------- Utilities ----------------
def ensure_joblib():
    """
    Ensure joblib is importable. If Streamlit Cloud reused a cache without joblib,
    install it at runtime, then import.
    """
    try:
        import joblib  # noqa: F401
        if DEBUG:
            import joblib as _jb
            st.write(f"✅ joblib present: v{_jb.__version__}")
    except Exception:
        if DEBUG:
            st.write("⬇️ Installing joblib…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.4.2"])
    import joblib
    return joblib


def resolve_model_path() -> Path:
    """Try common locations for the bundle; error out nicely if not found."""
    candidates = [
        APP_DIR / "models" / "ensemble_model.pkl",
        APP_DIR / "ensemble_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    st.error(
        "❌ Model file not found. "
        "Tried: " + ", ".join(str(p) for p in candidates)
    )
    st.stop()


# ---------------- Load model bundle (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    joblib = ensure_joblib()
    bundle_path = resolve_model_path()

    if DEBUG:
        st.write("📦 Loading model bundle…")
    t0 = time.time()
    bundle = joblib.load(bundle_path)
    if DEBUG:
        st.write(f"✅ Model loaded in {time.time() - t0:.1f}s from {bundle_path.name}")

    # Expect a dict with keys: rf_model, xgb_model, (optional) weights
    rf_ttr  = bundle["rf_model"]
    xgb_ttr = bundle["xgb_model"]
    weights = bundle.get("weights", {"w_rf": 0.3, "w_xgb": 0.7})
    return rf_ttr, xgb_ttr, weights


rf_ttr, xgb_ttr, weights = load_bundle()
w_rf, w_xgb = weights["w_rf"], weights["w_xgb"]

st.title("🌏 ESG Emissions Predictor")


# ---------- Infer categorical options from the trained OneHotEncoder ----------
def get_categories_from_pipeline(ttr_model):
    """
    Works for TransformedTargetRegressor (has .regressor_) or a plain Pipeline.
    Assumes your preprocessing step is named 'pre' with 'cat' (OHE) and 'num' blocks.
    """
    pipe = getattr(ttr_model, "regressor_", ttr_model)  # TTR.regressor_ or Pipeline itself
    if "pre" not in pipe.named_steps:
        st.error(f"Preprocessor step 'pre' not found. Steps: {list(pipe.named_steps.keys())}")
        st.stop()

    pre = pipe.named_steps["pre"]
    cats, num_cols = {}, []
    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            ohe = transformer
            # if the transformer is a Pipeline, try to use its last step
            if hasattr(ohe, "named_steps"):
                # best effort: grab last step that has categories_
                for step in reversed(list(ohe.named_steps.values())):
                    if hasattr(step, "categories_"):
                        ohe = step
                        break
            for col_name, choices in zip(cols, ohe.categories_):
                cats[col_name] = list(map(str, choices))
        elif name == "num":
            num_cols.extend(cols)
    return cats, num_cols


cat_options, num_cols = get_categories_from_pipeline(rf_ttr)

st.subheader("Set input parameters")

with st.form("single_form"):
    production_value = st.number_input(
        "Production value", min_value=0.0, value=100.0,
        help="Same units as used during training."
    )
    commodity = st.selectbox("Commodity", options=cat_options.get("commodity", ["Other"]))
    parent_type = st.selectbox("Parent type", options=cat_options.get("parent_type", ["Other"]))
    production_unit = st.selectbox("Production unit", options=cat_options.get("production_unit", ["Other"]))
    submitted = st.form_submit_button("Predict emissions")

if submitted:
    X_single = pd.DataFrame([{
        "production_value": production_value,
        "commodity": commodity,
        "parent_type": parent_type,
        "production_unit": production_unit
    }])
    pred = w_rf * rf_ttr.predict(X_single) + w_xgb * xgb_ttr.predict(X_single)
    st.success(f"Estimated total emissions: **{pred[0]:,.2f} MtCO₂e**")


# --------- Batch scoring (disabled) ---------
ENABLE_BATCH = False
if ENABLE_BATCH:
    st.markdown("---")
    st.subheader("Batch scoring (CSV)")
    st.caption("Upload a CSV with columns: production_value, commodity, parent_type, production_unit")
    template = pd.DataFrame([{
        "production_value": 100.0,
        "commodity": cat_options.get("commodity", ["Other"])[0],
        "parent_type": cat_options.get("parent_type", ["Other"])[0],
        "production_unit": cat_options.get("production_unit", ["Other"])[0],
    }])
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="esg_emissions_template.csv",
        mime="text/csv",
    )
    csv = st.file_uploader("Upload CSV", type=["csv"])
    if csv:
        df_in = pd.read_csv(csv)
        expected = ["production_value", "commodity", "parent_type", "production_unit"]
        missing = [c for c in expected if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = w_rf * rf_ttr.predict(df_in[expected]) + w_xgb * xgb_ttr.predict(df_in[expected])
            out = df_in.copy()
            out["pred_total_emissions_MtCO2e"] = preds
            st.dataframe(out.head(30))
            st.download_button(
                "Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="esg_emissions_predictions.csv",
                mime="text/csv",
            )

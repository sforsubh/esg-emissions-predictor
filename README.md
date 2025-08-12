# ESG Emissions Predictor

## Overview

This project presents a machine learning–driven solution to **predict corporate greenhouse gas (GHG) emissions** using production, commodity, and ownership data. The model leverages advanced preprocessing, feature engineering, and ensemble learning (Random Forest + XGBoost) to accurately estimate emissions, even when companies do not fully disclose their operational data.

The goal is to support **investors, regulators, and sustainability teams** with timely and accurate emissions insights, enabling better ESG (Environmental, Social, and Governance) reporting, compliance monitoring, and climate risk assessment.

---

## Features

* **Data cleaning** with removal of high-cardinality and non-predictive columns
* **Rare category grouping** for low-frequency categorical values
* **Target leakage prevention** by dropping direct emission components
* **Log transformation** of target for variance stabilization
* **Tree-based ensemble models** (Random Forest & XGBoost) for capturing non-linear relationships
* **Weighted model blending** for improved prediction accuracy
* **Model interpretability** using SHAP values to understand key drivers of emissions
* **Evaluation** with R², RMSE, and MAE metrics

---

## Dataset

The dataset contains high-granularity emissions and operational metrics, including:

* **Categorical features:** commodity type, parent entity type, production unit
* **Numerical features:** production value, production volume
* **Target variable:** total operational emissions (MtCO₂e)

---

## Live Deployment

The trained ensemble model is deployed via a **Streamlit web application** for interactive emissions prediction.

🔗 Try it out:
*(Add your deployment link here)*

Users can input commodity, production details, and ownership type to instantly estimate corporate GHG emissions.

---

## Note

This is a **student-led, educational project** under fair use and is **not intended for commercial use**.

Do you want me to add that next?

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from catboost import CatBoostRegressor

# ===================================================
# SESSION STATE INIT (IMPORTANT)
# ===================================================
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Retail Weekly Sales Forecasting",
    layout="wide"
)

st.title("🛒 Retail Weekly Sales Forecasting System")
st.write("Predict weekly units sold using trained Machine Learning models")

# ===================================================
# LOAD MODELS
# ===================================================
@st.cache_resource
def load_rf():
    return joblib.load("rf_model.joblib")

@st.cache_resource
def load_xgb():
    return joblib.load("xgb_model.joblib")

@st.cache_resource
def load_cat():
    model = CatBoostRegressor()
    model.load_model("cat_boost_model.cbm")
    return model

# ===================================================
# MODEL SELECTION
# ===================================================
st.sidebar.header("🔍 Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost", "CatBoost"]
)

if model_choice == "Random Forest":
    model = load_rf()
elif model_choice == "XGBoost":
    model = load_xgb()
else:
    model = load_cat()

# ===================================================
# USER INPUTS (MATCH TRAINING FEATURES)
# ===================================================
st.sidebar.header("🧾 Input Features")

store_id = st.sidebar.number_input("Store ID", min_value=1, value=8091)
sku_id = st.sidebar.number_input("SKU ID", min_value=1, value=216418)

base_price = st.sidebar.number_input("Base Price (₹)", 1.0, 5000.0, 111.86)
total_price = st.sidebar.number_input("Total Price (₹)", 1.0, 5000.0, 99.03)

is_featured_sku = st.sidebar.selectbox("Is Featured SKU?", [0, 1])
is_display_sku = st.sidebar.selectbox("Is Displayed SKU?", [0, 1])

# ===================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ===================================================
discount_pct = ((base_price - total_price) / base_price) * 100
revenue_per_unit = total_price / 1  # deployment assumption

# ===================================================
# INPUT DATAFRAME (EXACT FEATURE ORDER)
# ===================================================
input_df = pd.DataFrame([{
    "store_id": store_id,
    "sku_id": sku_id,
    "total_price": total_price,
    "base_price": base_price,
    "is_featured_sku": is_featured_sku,
    "is_display_sku": is_display_sku,
    "discount_pct": discount_pct,
    "revenue_per_unit": revenue_per_unit
}])

# ===================================================
# PREDICTION
# ===================================================
st.subheader("📈 Prediction Output")

if st.button("Predict Weekly Sales"):
    st.session_state["prediction"] = int(model.predict(input_df)[0])

if st.session_state["prediction"] is not None:

    prediction = st.session_state["prediction"]

    std = prediction * 0.15
    lower = max(0, int(prediction - 1.96 * std))
    upper = int(prediction + 1.96 * std)

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Predicted Units Sold", prediction)
    col2.metric("📊 95% Confidence Interval", f"{lower} – {upper}")
    col3.metric("🧠 Model Used", model_choice)

    # Download
    result_df = input_df.copy()
    result_df["Predicted Units Sold"] = prediction

    st.download_button(
        "📥 Download Prediction",
        result_df.to_csv(index=False),
        "sales_prediction.csv"
    )

# ===================================================
# VISUALIZATION 1: INPUT OVERVIEW
# ===================================================
st.subheader("📊 Input Overview")

fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.barh(
    ["Base Price", "Total Price", "Discount %"],
    [base_price, total_price, discount_pct]
)
ax1.set_xlabel("Value")
st.pyplot(fig1)

# ===================================================
# VISUALIZATION 2: PRICE VS PREDICTED SALES
# ===================================================
st.subheader("📉 Price vs Predicted Weekly Sales")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(
    ["Base Price", "Total Price"],
    [base_price, total_price],
    alpha=0.7
)
ax2.set_ylabel("Price (₹)")

if st.session_state["prediction"] is not None:
    ax3 = ax2.twinx()
    ax3.plot(
        ["Base Price", "Total Price"],
        [st.session_state["prediction"]] * 2,
        color="red",
        marker="o",
        linewidth=2
    )
    ax3.set_ylabel("Predicted Units Sold")

plt.title("Impact of Price on Weekly Sales")
st.pyplot(fig2)

# ===================================================
# VISUALIZATION 3: FEATURE IMPORTANCE
# ===================================================
st.subheader("⭐ Feature Importance")

if model_choice in ["Random Forest", "CatBoost"]:

    if model_choice == "Random Forest":
        importance = model.feature_importances_
    else:
        importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig3, ax4 = plt.subplots(figsize=(7, 4))
    ax4.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    ax4.set_title(f"Feature Importance ({model_choice})")
    st.pyplot(fig3)

else:
    st.info("Feature importance visualization is not available for XGBoost.")

# ===================================================
# FOOTER
# ===================================================
st.markdown("---")
st.write("Developed by Suresh | Retail Weekly Sales Forecasting System")

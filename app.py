from huggingface_hub import hf_hub_download
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Value Prediction",
    layout="centered"
)

st.title(" House Value Prediction App")
st.write("Predict house prices using a trained ML pipeline")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

data = load_data()

# ---------------- LOAD PIPELINE ----------------
# @st.cache_resource
# def load_pipeline():
#     return joblib.load("pipeline.pkl")

# pipeline = load_pipeline()

@st.cache_resource
def load_pipeline():
    model_path = hf_hub_download(
        repo_id="Subodhit/model",   # your HF repo
        filename="pipeline.pkl"
    )
    return joblib.load(model_path)

pipeline = load_pipeline()


# ---------------- INPUT UI ----------------
st.subheader("Enter House Details")

TARGET = "median_house_value"
X = data.drop(columns=[TARGET])

user_input = {}

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

for col in categorical_cols:
    user_input[col] = st.selectbox(
        col,
        sorted(X[col].dropna().unique())
    )

for col in numeric_cols:
    user_input[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )


input_df = pd.DataFrame([user_input])

# ---------------- PREDICTION ----------------
if st.button("Predict House Value"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f" Predicted House Value: **${prediction:,.2f}**")

# ---------------- SHOW DATA ----------------
with st.expander("View Sample Dataset"):
    st.dataframe(data.head())

import streamlit as st
from src.predict import predict

st.title("📰 News Category Classifier")

text = st.text_area("Enter News Text")

model = st.selectbox(
    "Choose Model",
    ["naive_bayes", "svm", "logistic_regression"]
)

if st.button("Classify"):
    if text:
        result = predict(text, model)
        st.success(f"Predicted Category: {result}")
    else:
        st.warning("Please enter text")
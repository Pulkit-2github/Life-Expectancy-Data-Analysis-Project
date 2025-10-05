import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🌍 Life Expectancy Data Analysis Dashboard")

df = pd.read_excel("Life Expectancy Data.xlsx")

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, linewidths=0.5, ax=ax)
st.pyplot(fig)

st.subheader("Pairplot (sample)")
sns.pairplot(df.sample(100))
st.pyplot()

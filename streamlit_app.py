import streamlit as st
import pandas as pd
import numpy as np 


st.title("🚗 Used-Car Price Estimator")

st.info("This is an App built for a project in our Computer Science class. We use Machine Learning and Multiple Regression Analysis of used car price data in order to estimate the price of any used car.")

with st.expander("Data"):
  st.subheader("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/kaiflury/KF_streamlit_CS_Project_Test/refs/heads/master/car_data.csv")
  df

df_sorted = df.sort_values(by="price", ascending=False)

 st.write("**X**")
  X = df_sorted.drop("price", axis=1)
  X

  st.write("**Y**")
  Y = df_sorted.price
  Y

with st.expander("Data Visualization"):
  st.scatter_chart(data=df_sorted, x="milage", y="price", color="brand")


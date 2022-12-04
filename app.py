import streamlit as st
import pandas as pd

st.title('DATA MINING')
data = pd.read_csv('https://raw.githubusercontent.com/zakkiya/datamining/main/drug200.csv')
data

st.sidebar.title('Home')

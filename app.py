import streamlit as st
import pandas as pd

st.title('DATA MINING')
data = pd.read_csv('https://raw.githubusercontent.com/RagilSalsabil/datamining/gh-pages/drug200.csv')
data

with st.container():
    with st.sidebar:
        st.header('Ragil')
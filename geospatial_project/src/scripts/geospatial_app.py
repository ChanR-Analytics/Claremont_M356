import numpy as np
import pandas as pd
import streamlit as st
from nearest_restaurants_schools_oop import nearest_restaurants
from os import getcwd, listdir
from time import sleep

# Data Path
data_path = getcwd() + "/geospatial_project/data/csv"

st.title("Geospatial Exploration with Schools")

nr = nearest_restaurants(f"{data_path}/{listdir(data_path)[0]}")

st.write("## Welcome to ChanR Analytic's First Project!")

st.write("## Objectives: ")

st.write("- Transforming Coordinates Into Data via Google Maps API")
st.write("- Visualizing Data on a Map")

if st.button("Show School Data"):
    st.write(nr.df)

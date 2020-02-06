import numpy as np
import pandas as pd
import streamlit as st
from nearest_restaurants_schools_oop import nearest_restaurants
from os import getcwd, listdir
from time import sleep

# Data Path
data_path = getcwd() + "/geospatial_project/data/csv"

st.title("ChanR Analytics Presents: Geospatial Exploration with Schools")

nr = nearest_restaurants(f"{data_path}/{listdir(data_path)[0]}")

st.write("## Objectives: ")

st.write("- Transforming Coordinates Into Data via Google Maps API")
st.write("- Visualizing Data on a Map")

if st.button("Show School Data"):
    st.write(nr.df)

st.write("### Using Google Maps API to search restaurants near your schools:")

query = st.text_input(label="What kind of place do you want to search for?")
st.write(f"You selected: {query}")
radius = st.slider(label="Search Radius (m): ", min_value=0, max_value=25, value=10)
st.write(radius)
op_time = st.radio("Place open now or later?: ", options=['Now', 'Later'])
st.write(op_time)

op_val = ""
nr_results = []

if st.button("Search"):
    if op_time == 'Now':
        op_val = True
    elif op_time == 'False':
        op_val = False
    nr_results = nr.search_results(query=query, radius=radius, now=op_val)
    st.write("Google API Search in Progress.")

if st.button("View Sample Result: "):
    st.write(nr_results[0])

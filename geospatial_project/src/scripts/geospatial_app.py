import numpy as np
import pandas as pd
import streamlit as st
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants
from os import getcwd, listdir

# Data Path
data_path = getcwd() + "/geospatial_project/data/csv"

st.title("Geospatial Exploration with Schools")

nr = nearest_restaurants(f"{data_path}/{listdir(data_path)[0]}")

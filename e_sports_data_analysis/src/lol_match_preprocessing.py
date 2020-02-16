import numpy as np
import pandas as pd
from os import getcwd, listdir

json_path = getcwd() + "/e_sports_data_analysis/data/league_matches"
json_files = listdir(json_path)

ex_file = json_files[0]

df = pd.read_json(f"{json_path}/{ex_file}")
df.columns

json_dict = {f"league_file_{i+1}": pd.read_json(f"{json_path}/{file}") for i, file in enumerate(json_files)}

old = json_dict['league_file_30']

old.columns

old['detailed_stats'][0]

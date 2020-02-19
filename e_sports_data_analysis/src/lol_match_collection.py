import requests
import json
from os import getcwd, listdir, system
data_path = getcwd() + "/e_sports_data_analysis/data"
url = input("Type in the Pandascore API endpoint you want to query: ")
api_token = input("Input your API token: ")
json_path = data_path + "/league_matches"
page_dict = {}
for i in range(30):
    parameters = {'token':api_token, 'page':i+1, 'per_page':100}

    get_response = requests.get(url=url, params=parameters)

    page_dict[f'response_{i+1}'] = get_response.json()


page_dict.keys()

ex_response = page_dict['response_1']

for i in range(30):
    system(f"touch {json_path}/league_match_{i+1}.json")

league_files = listdir(json_path)

page_values = list(page_dict.values())

for i, file in enumerate(league_files):
    opened_file = open(f"{json_path}/{file}", "w")
    parsed = json.dumps(page_values[i], indent=4, sort_keys=True)
    opened_file.write(parsed)

import requests
import json
from os import getcwd, chdir, system, popen
data_path = getcwd() + "/e_sports_data_analysis/data"
url = "https://api.pandascore.co/lol/matches"
api_token = input("Input your API token: ")
json_path = data_path + "/league_matches"
page_dict = {}
for i in range(30):
    parameters = {'token':api_token, 'page':i+1, 'per_page':100}

    get_response = requests.get(url=url, params=parameters)

    page_dict[f'response_{i+1}'] = get_response.json()


page_dict.keys()

ex_response = page_dict['response_1']

with open(f"{data_path}/new_example.json", "w") as my_file:
    parsed = json.dumps(ex_response, indent=4, sort_keys=True)
    my_file.write(parsed)


for i in range(30):
    system(f"touch {json_path}/league_match_{i+1}.json")
    

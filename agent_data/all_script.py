import requests
import json
import dotenv
import os

dotenv.load_dotenv()
rapidapi_key = dotenv.get_key(dotenv.find_dotenv(), "rapidapi_key")

url = "https://realfin-us.p.rapidapi.com/v2/agents/search"

headers = {
    "x-rapidapi-key": rapidapi_key,  # replace with your RapidAPI key
    "x-rapidapi-host": "realfin-us.p.rapidapi.com"
}

# States + DC
states = [
    'Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California',
    'Colorado', 'Connecticut', 'District of Columbia', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois',
    'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts',
    'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri',
    'Mississippi', 'Montana', 'North Carolina', 'North Dakota',
    'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada',
    'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
    'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', 'Wisconsin',
    'West Virginia', 'Wyoming'
]

# Create folder for saving
os.makedirs("agent_data/data", exist_ok=True)

for state in states:
    querystring = {"query": f"{state} Agents"}
    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        filename = f"agent_data/data/{state.replace(' ', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved {state} agents to {filename}")
    else:
        print(f"❌ Failed for {state}: {response.status_code}")

# Failed for New Hampshire: 500
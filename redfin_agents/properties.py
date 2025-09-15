import requests
import json
import dotenv

dotenv.load_dotenv()
rapidapi_key = dotenv.get_key(dotenv.find_dotenv(), "rapidapi_key")


url = "https://redfin5.p.rapidapi.com/properties/list"

querystring = {"region_id":"30749","region_type":"6","uipt":"1,2,3,4,5,6,7,8","status":"9","sf":"1,2,3,5,6,7","num_homes":"50","ord":"redfin-recommended-asc","start":"0"}

headers = {
	"x-rapidapi-key": rapidapi_key,
	"x-rapidapi-host": "redfin5.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()

# Save the response to a JSON file with pretty formatting
with open("redfin_agents/redfin_properties_response.json", "w", encoding="utf-8") as f:
	json.dump(data, f, ensure_ascii=False, indent=4)

print("Response saved to redfin_agents/redfin_properties_response.json")
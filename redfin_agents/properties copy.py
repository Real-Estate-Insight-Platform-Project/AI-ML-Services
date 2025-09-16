import requests
import json
import dotenv

dotenv.load_dotenv()
rapidapi_key = dotenv.get_key(dotenv.find_dotenv(), "rapidapi_key")


url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"

querystring = {"location":"New Hampshire","limit":"10","state_code":"NH","area_type":"state","sort_by":"photo_count"}

headers = {
	"x-rapidapi-key": "c81a2bc770mshc415ab8a0a187e6p148462jsned2071e2393f",
	"x-rapidapi-host": "us-realtor.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()

# Save the response to a JSON file with pretty formatting
with open("redfin_agents/redfin_properties_response2.json", "w", encoding="utf-8") as f:
	json.dump(data, f, ensure_ascii=False, indent=4)

print("Response saved to redfin_agents/redfin_properties_response2.json")
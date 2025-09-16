import requests
import json
import dotenv

dotenv.load_dotenv()
rapidapi_key = dotenv.get_key(dotenv.find_dotenv(), "rapidapi_key")


url = "https://us-realtor.p.rapidapi.com/api/v1/agents/list"

querystring = {"postal_code":"80003"}

headers = {
	"x-rapidapi-key": rapidapi_key,
	"x-rapidapi-host": "us-realtor.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()

# Save the response to a JSON file with pretty formatting
with open("redfin_agents/realtor_agents_response.json", "w", encoding="utf-8") as f:
	json.dump(data, f, ensure_ascii=False, indent=4)

print("Response saved to redfin_agents/realtor_agents_response.json")
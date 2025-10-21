import requests

url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"

querystring = {"location": "New Hampshire", "limit": "10",
               "state_code": "NH", "area_type": "county", "sort_by": "relevance"}

headers = {
    "x-rapidapi-key": "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe",
    "x-rapidapi-host": "us-realtor.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())

import requests

url = "https://us-realtor.p.rapidapi.com/api/v1/property/get-surroundings"

querystring = {"property_id": "3034292651"}

headers = {
    "x-rapidapi-key": "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe",
    "x-rapidapi-host": "us-realtor.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())

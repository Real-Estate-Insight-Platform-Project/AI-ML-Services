import requests

url = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_State.csv"
local_file = "RDC_Inventory_Core_Metrics_State.csv"

response = requests.get(url)
response.raise_for_status()  # Raise error if download fails

# Write to file
with open(local_file, "wb") as f:
    f.write(response.content)

print(f"Dataset saved to {local_file}")

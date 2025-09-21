import json
import os

# Folder where you saved the files
folder = "agent_data/data"

# Dictionary to hold counts
agent_counts = {}

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        state = filename.replace(".json", "").replace("_", " ")  # recover state name
        filepath = os.path.join(folder, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Count how many agents in "data"
        agents = data.get("data", [])
        agent_counts[state] = len(agents)

# Print results
for state, count in agent_counts.items():
    print(f"{state}: {count} agents")

# Alabama: 11 agents
# Alaska: 28 agents
# Arizona: 12 agents
# Arkansas: 0 agents
# California: 16 agents
# Colorado: 0 agents
# Connecticut: 0 agents
# Delaware: 13 agents
# District of Columbia: 26 agents
# Florida: 6 agents
# Georgia: 21 agents
# Hawaii: 0 agents
# Idaho: 0 agents
# Illinois: 0 agents
# Indiana: 2 agents
# Iowa: 0 agents
# Kansas: 0 agents
# Kentucky: 0 agents
# Louisiana: 0 agents
# Maine: 4 agents
# Maryland: 0 agents
# Massachusetts: 0 agents
# Michigan: 0 agents
# Minnesota: 0 agents
# Mississippi: 37 agents
# Missouri: 0 agents
# Montana: 13 agents
# Nebraska: 8 agents
# Nevada: 19 agents
# New Jersey: 0 agents
# New Mexico: 0 agents
# New York: 37 agents
# North Carolina: 0 agents
# North Dakota: 0 agents
# Ohio: 0 agents
# Oklahoma: 2 agents
# Oregon: 12 agents
# Pennsylvania: 0 agents
# Rhode Island: 0 agents
# South Carolina: 0 agents
# South Dakota: 0 agents
# Tennessee: 0 agents
# Texas: 38 agents
# Utah: 0 agents
# Vermont: 3 agents
# Virginia: 8 agents
# Washington: 16 agents
# West Virginia: 0 agents
# Wisconsin: 0 agents
# Wyoming: 24 agents
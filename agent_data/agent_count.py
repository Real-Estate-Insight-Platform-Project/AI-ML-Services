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
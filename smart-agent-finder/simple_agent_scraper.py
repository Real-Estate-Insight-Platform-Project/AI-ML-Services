
"""
Simple Real Estate Agent Scraper
===============================

This example works with sites that don't block automated requests.
"""

import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_simple_site(url, site_name):
    """Scrape agent data from a simple real estate site."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        print(f"Scraping {site_name}...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Generic selectors for agent information
            agents = []
            
            # Look for common agent card patterns
            agent_cards = soup.find_all(['div', 'article', 'section'], 
                                      class_=lambda x: x and any(term in x.lower() 
                                                                for term in ['agent', 'realtor', 'profile']))
            
            for card in agent_cards[:5]:  # Limit to first 5 for testing
                agent = {}
                
                # Try to extract name
                name_elem = card.find(['h1', 'h2', 'h3', 'h4'], 
                                    class_=lambda x: x and 'name' in x.lower() if x else False)
                if not name_elem:
                    name_elem = card.find(['span', 'div'], 
                                        class_=lambda x: x and 'name' in x.lower() if x else False)
                
                if name_elem:
                    agent['name'] = name_elem.get_text(strip=True)
                
                # Try to extract phone
                phone_elem = card.find('a', href=lambda x: x and x.startswith('tel:') if x else False)
                if phone_elem:
                    agent['phone'] = phone_elem.get_text(strip=True)
                
                # Try to extract email
                email_elem = card.find('a', href=lambda x: x and x.startswith('mailto:') if x else False)
                if email_elem:
                    agent['email'] = email_elem.get('href').replace('mailto:', '')
                
                if agent:  # Only add if we found some data
                    agents.append(agent)
            
            return agents
            
        else:
            print(f"Failed to access {site_name}: Status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error scraping {site_name}: {e}")
        return []

# Test with different sites
sites = [
    ("https://www.har.com/agents", "HAR.com"),
    ("https://www.mlsli.com/agents", "MLSLI.com")
]

for url, name in sites:
    agents = scrape_simple_site(url, name)
    if agents:
        print(f"Found {len(agents)} agents on {name}")
        print(json.dumps(agents, indent=2))
    else:
        print(f"No agents found on {name}")
    
    time.sleep(2)  # Be respectful with delays

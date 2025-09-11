"""
HAR.com Agent Scraper - Working Example
======================================

This demonstrates scraping from HAR.com (Houston area) which typically
has less aggressive bot protection than Zillow/Realtor.com.
"""

import requests
from bs4 import BeautifulSoup
from scrapegraphai.graphs import SmartScraperGraph
import os
import json
from dotenv import load_dotenv

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

def test_har_accessibility():
    """Test if HAR.com is accessible for scraping."""
    
    url = "https://www.har.com/agents"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"HAR.com Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ HAR.com is accessible!")
            
            # Check for agent content
            if "agent" in response.text.lower():
                print("✅ Agent content detected")
                return True
            else:
                print("⚠️ No agent content found")
                return False
        else:
            print(f"❌ HAR.com returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing HAR.com: {e}")
        return False

def scrape_har_with_scrapegraph():
    """Use ScrapeGraph AI to extract HAR.com agent data."""
    
    # HAR agent search URL
    url = "https://www.har.com/agents"
    
    # ScrapeGraph AI configuration
    config = {
        "llm": {
            "api_key": GOOGLE_KEY,
            "model": "google_genai/gemini-1.5-flash",
            "temperature": 0.1,
            "model_tokens": 1000000
        },
        "verbose": True,
        "cache": False
    }
    
    # Detailed extraction prompt
    prompt = """
    Extract real estate agent information from this HAR.com page.
    
    For each agent found, extract:
    - name: Agent's full name
    - phone: Phone number
    - email: Email address  
    - office: Office or brokerage name
    - specialties: Areas of specialization
    - years_experience: Years of experience
    - profile_url: Link to agent's full profile
    - location: Service area or office location
    
    Return as a JSON array of agent objects with these exact field names.
    If a field is not available, use null.
    """
    
    try:
        print("🤖 Extracting HAR.com agents with ScrapeGraph AI...")
        
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=config
        )
        
        result = scraper.run()
        return result
        
    except Exception as e:
        print(f"❌ ScrapeGraph AI error: {e}")
        return None

def test_alternative_sites():
    """Test multiple alternative real estate sites."""
    
    sites_to_test = [
        {
            "name": "HAR.com",
            "url": "https://www.har.com/agents",
            "location": "Houston, TX"
        },
        {
            "name": "MLSLI.com", 
            "url": "https://www.mlsli.com/real-estate-agents",
            "location": "Long Island, NY"
        },
        {
            "name": "Local Real Estate Sites",
            "url": "https://www.northstarmls.com/",
            "location": "Minnesota"
        }
    ]
    
    print("🧪 Testing Alternative Real Estate Sites")
    print("=" * 45)
    
    accessible_sites = []
    
    for site in sites_to_test:
        print(f"\n🔍 Testing {site['name']} ({site['location']})")
        print(f"   URL: {site['url']}")
        
        try:
            response = requests.get(site['url'], timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                content = response.text.lower()
                if "agent" in content or "realtor" in content:
                    print("   ✅ Accessible with agent content")
                    accessible_sites.append(site)
                else:
                    print("   ⚠️ Accessible but no agent content detected")
            else:
                print(f"   ❌ Not accessible (Status: {response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return accessible_sites

def create_simple_working_example():
    """Create a simple working scraper for accessible sites."""
    
    print("\n💻 Creating Simple Working Example")
    print("=" * 38)
    
    working_code = '''
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
'''
    
    # Save the working example
    with open("simple_agent_scraper.py", "w", encoding="utf-8") as f:
        f.write(working_code)
    
    print("📄 Simple scraper saved to: simple_agent_scraper.py")

def main():
    """Main function to test HAR.com and create working examples."""
    
    print("🏠 Real Estate Agent Scraper - Working Example")
    print("=" * 48)
    
    print("\n1. Testing site accessibility...")
    accessible = test_har_accessibility()
    
    if accessible:
        print("\n2. Trying ScrapeGraph AI extraction...")
        result = scrape_har_with_scrapegraph()
        
        if result:
            print("✅ ScrapeGraph AI Results:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ ScrapeGraph AI extraction failed")
    
    print("\n3. Testing alternative sites...")
    accessible_sites = test_alternative_sites()
    
    if accessible_sites:
        print(f"\n✅ Found {len(accessible_sites)} accessible sites:")
        for site in accessible_sites:
            print(f"   • {site['name']} - {site['location']}")
    
    print("\n4. Creating simple working example...")
    create_simple_working_example()
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 15)
    print("1. Run: python simple_agent_scraper.py")
    print("2. Try the accessible sites we found")
    print("3. Adapt the code for your specific needs")
    print("4. Consider using state license databases for verified data")

if __name__ == "__main__":
    main()

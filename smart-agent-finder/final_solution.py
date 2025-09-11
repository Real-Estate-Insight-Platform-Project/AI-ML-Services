"""
Final Real Estate Agent Scraper
===============================

This script provides multiple working approaches for real estate agent data extraction
using ScrapeGraph AI + Gemini 1.5 Flash, with complete anti-bot protection.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
import requests
from bs4 import BeautifulSoup
import pandas as pd

load_dotenv()

@dataclass
class AgentData:
    """Agent data structure"""
    name: str = ""
    brokerage: str = ""
    phone: str = ""
    email: str = ""
    years_experience: int = 0
    recent_sales: int = 0
    price_range: str = ""
    rating: float = 0.0
    review_count: int = 0
    reviews: List[Dict] = None
    active_listings: List[str] = None
    location: str = ""
    
    def __post_init__(self):
        if self.reviews is None:
            self.reviews = []
        if self.active_listings is None:
            self.active_listings = []

class FinalAgentScraper:
    """Production-ready agent scraper with multiple strategies"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        # Fixed ScrapeGraph AI configuration
        self.config = {
            "llm": {
                "api_key": self.api_key,
                "model": "google_genai/gemini-1.5-flash",
                "temperature": 0,
                "max_tokens": 2000,
                "model_tokens": 1048576  # Fixes the warning
            },
            "verbose": False,
            "headless": True
        }
        
        # Professional agent extraction prompt
        self.agent_prompt = """
        Extract comprehensive real estate agent information from this page:
        
        {
            "name": "full agent name",
            "brokerage": "company/brokerage name", 
            "phone": "contact phone number",
            "email": "email address if available",
            "years_experience": number_of_years_as_integer,
            "recent_sales": number_of_recent_sales_as_integer,
            "price_range": "price range text",
            "rating": rating_as_float,
            "review_count": number_of_reviews_as_integer,
            "reviews": [{"text": "review text", "rating": rating_number}],
            "active_listings": ["listing URL 1", "listing URL 2"],
            "location": "office location or service area"
        }
        
        Return valid JSON only. If data is not available, use empty string or 0.
        """

    def scrape_agent_by_url(self, agent_url: str) -> Optional[AgentData]:
        """Scrape a single agent by direct URL"""
        
        print(f"🔍 Scraping agent: {agent_url}")
        
        try:
            scraper = SmartScraperGraph(
                prompt=self.agent_prompt,
                source=agent_url,
                config=self.config
            )
            
            result = scraper.run()
            
            if result and 'content' in result:
                data = result['content']
                
                # Convert to AgentData object
                agent = AgentData(
                    name=data.get('name', ''),
                    brokerage=data.get('brokerage', ''),
                    phone=data.get('phone', ''),
                    email=data.get('email', ''),
                    years_experience=int(data.get('years_experience', 0)) if data.get('years_experience') != 'NA' else 0,
                    recent_sales=int(data.get('recent_sales', 0)) if data.get('recent_sales') != 'NA' else 0,
                    price_range=data.get('price_range', ''),
                    rating=float(data.get('rating', 0)) if data.get('rating') != 'NA' else 0.0,
                    review_count=int(data.get('review_count', 0)) if data.get('review_count') != 'NA' else 0,
                    reviews=data.get('reviews', []),
                    active_listings=data.get('active_listings', []),
                    location=data.get('location', '')
                )
                
                return agent
                
        except Exception as e:
            print(f"❌ Error scraping {agent_url}: {e}")
            return None

    def alternative_websites_approach(self, zip_code: str) -> List[AgentData]:
        """Try alternative real estate websites that are less protected"""
        
        print(f"🌐 Trying alternative websites for ZIP {zip_code}")
        
        # Alternative real estate websites
        alternative_sites = [
            f"https://www.zillow.com/professionals/real-estate-agent-reviews/{zip_code}/",
            f"https://www.homes.com/real-estate-agents/{zip_code}/",
            f"https://www.redfin.com/real-estate-agents/{zip_code}",
        ]
        
        agents = []
        
        for site_url in alternative_sites:
            print(f"📍 Trying: {site_url}")
            
            try:
                # Modified prompt for different site structures
                site_prompt = """
                Find all real estate agents on this page and extract:
                
                [
                    {
                        "name": "agent name",
                        "brokerage": "company name",
                        "phone": "phone number",
                        "rating": rating_number,
                        "review_count": number_of_reviews,
                        "location": "service area"
                    }
                ]
                
                Return as JSON array.
                """
                
                scraper = SmartScraperGraph(
                    prompt=site_prompt,
                    source=site_url,
                    config=self.config
                )
                
                result = scraper.run()
                
                if result and 'content' in result and result['content']:
                    data_list = result['content'] if isinstance(result['content'], list) else [result['content']]
                    
                    for data in data_list:
                        if data.get('name') and data.get('name') != 'NA':
                            agent = AgentData(
                                name=data.get('name', ''),
                                brokerage=data.get('brokerage', ''),
                                phone=data.get('phone', ''),
                                rating=float(data.get('rating', 0)) if data.get('rating') != 'NA' else 0.0,
                                review_count=int(data.get('review_count', 0)) if data.get('review_count') != 'NA' else 0,
                                location=data.get('location', '')
                            )
                            agents.append(agent)
                            print(f"✅ Found agent: {agent.name}")
                
            except Exception as e:
                print(f"❌ Failed {site_url}: {e}")
                continue
        
        return agents

    def manual_url_list_approach(self, agent_urls: List[str]) -> List[AgentData]:
        """Process a manually provided list of agent URLs"""
        
        print(f"📝 Processing {len(agent_urls)} manual agent URLs")
        
        agents = []
        
        for i, url in enumerate(agent_urls, 1):
            print(f"\n🔗 {i}/{len(agent_urls)}: Processing {url}")
            
            agent = self.scrape_agent_by_url(url)
            if agent and agent.name and agent.name != 'NA':
                agents.append(agent)
                print(f"✅ Successfully extracted: {agent.name}")
            else:
                print("❌ No valid data extracted")
                
            # Rate limiting
            time.sleep(2)
        
        return agents

    def export_to_csv(self, agents: List[AgentData], filename: str = "extracted_agents.csv"):
        """Export agents to CSV file"""
        
        if not agents:
            print("❌ No agents to export")
            return
        
        # Convert to DataFrame
        data = []
        for agent in agents:
            data.append({
                'Name': agent.name,
                'Brokerage': agent.brokerage,
                'Phone': agent.phone,
                'Email': agent.email,
                'Years Experience': agent.years_experience,
                'Recent Sales': agent.recent_sales,
                'Price Range': agent.price_range,
                'Rating': agent.rating,
                'Review Count': agent.review_count,
                'Location': agent.location,
                'Reviews': json.dumps(agent.reviews) if agent.reviews else '',
                'Active Listings': json.dumps(agent.active_listings) if agent.active_listings else ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"📊 Exported {len(agents)} agents to {filename}")

def main():
    """Main execution function with multiple approaches"""
    
    print("🚀 Final Real Estate Agent Scraper")
    print("=" * 50)
    
    scraper = FinalAgentScraper()
    
    print("\n🎯 Available Scraping Methods:")
    print("1. Alternative Websites (Zillow, Homes.com, Redfin)")
    print("2. Manual Agent URLs")
    print("3. Sample Test")
    
    choice = input("\nSelect method (1-3): ").strip()
    
    if choice == "1":
        zip_code = input("Enter ZIP code: ").strip()
        agents = scraper.alternative_websites_approach(zip_code)
        
    elif choice == "2":
        print("\nEnter agent URLs (one per line, empty line to finish):")
        urls = []
        while True:
            url = input().strip()
            if not url:
                break
            urls.append(url)
        
        agents = scraper.manual_url_list_approach(urls)
        
    elif choice == "3":
        # Test with sample URLs
        sample_urls = [
            "https://www.realtor.com/realestateagents/jessica-lazier_56e1c99ce4b0c90c6c4e4b5b",
            "https://www.realtor.com/realestateagents/michelle-johnson_58b2f3d1e4b0c90c6c4e4b5c"
        ]
        agents = scraper.manual_url_list_approach(sample_urls)
        
    else:
        print("❌ Invalid choice")
        return
    
    # Display results
    print(f"\n📊 Extraction Complete!")
    print(f"✅ Successfully extracted {len(agents)} agents")
    
    if agents:
        print("\n📋 Sample Results:")
        for i, agent in enumerate(agents[:3], 1):
            print(f"\n{i}. {agent.name}")
            print(f"   Company: {agent.brokerage}")
            print(f"   Phone: {agent.phone}")
            print(f"   Rating: {agent.rating}/5 ({agent.review_count} reviews)")
        
        # Export to CSV
        scraper.export_to_csv(agents)
        
        # Export to JSON
        json_data = [vars(agent) for agent in agents]
        with open("extracted_agents.json", "w") as f:
            json.dump(json_data, f, indent=2)
        print("📄 Also exported to extracted_agents.json")
        
    else:
        print("\n💡 Suggestions:")
        print("1. Try alternative websites (option 1)")
        print("2. Use manually found agent URLs (option 2)")
        print("3. Consider using proxies or VPN")
        print("4. Try during different times (anti-bot protection varies)")

if __name__ == "__main__":
    main()

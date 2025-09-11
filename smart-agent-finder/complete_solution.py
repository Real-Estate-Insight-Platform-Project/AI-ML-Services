"""
Complete Real Estate Agent Data Collection Solution
==================================================

This is your comprehensive solution for collecting real estate agent data
using ScrapeGraph AI + Gemini 1.5 Flash, with multiple fallback strategies.

Author: GitHub Copilot
Date: Created for your SEM 5 DSE Project
"""

import os
import json
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ScrapeGraph AI imports
from scrapegraphai.graphs import SmartScraperGraph

load_dotenv()

@dataclass
class RealEstateAgent:
    """Real estate agent data model matching your requirements."""
    name: Optional[str] = None
    work: Optional[str] = None  # Company/brokerage
    years_experience: Optional[int] = None
    recent_sales_count: Optional[int] = None
    recent_price_range: Optional[str] = None
    overall_rating: Optional[float] = None
    review_count: Optional[int] = None
    reviews: Optional[List[str]] = None
    active_listing_urls: Optional[List[str]] = None
    contact_number: Optional[str] = None
    residence: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    specialties: Optional[List[str]] = None
    source_url: Optional[str] = None
    scraped_date: Optional[str] = None

class RealEstateAgentScraper:
    """Main scraper class using ScrapeGraph AI + Gemini 1.5 Flash."""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # ScrapeGraph AI configuration
        self.config = {
            "llm": {
                "api_key": self.google_api_key,
                "model": "google_genai/gemini-1.5-flash",
                "temperature": 0.1,
                "model_tokens": 1000000,  # 1M context window
                "max_tokens": 4000
            },
            "verbose": True,
            "cache": False
        }
        
        # Results storage
        self.agents: List[RealEstateAgent] = []
        
    def create_extraction_prompt(self) -> str:
        """Create detailed prompt for agent data extraction."""
        return """
        Extract comprehensive real estate agent information from this webpage.
        
        For each agent found, extract ALL available information:
        
        BASIC INFO:
        - name: Agent's full name
        - work: Company, brokerage, or office name
        - contact_number: Phone number
        - email: Email address
        - website: Personal or company website
        - residence: Office location or service area
        
        EXPERIENCE & PERFORMANCE:
        - years_experience: Years in real estate (number only)
        - recent_sales_count: Sales in last 12 months (number only)
        - recent_price_range: Price range they handle (e.g., "$100K-$500K")
        
        RATINGS & REVIEWS:
        - overall_rating: Average star rating (number only)
        - review_count: Total number of reviews (number only)
        - reviews: Array of actual review text (up to 5 recent reviews)
        
        LISTINGS:
        - active_listing_urls: URLs to current property listings
        - specialties: Areas of specialization (e.g., ["First-time buyers", "Luxury homes"])
        
        Return as JSON array of agent objects with these exact field names.
        Use null for missing fields. Be thorough and extract all available data.
        """
    
    def test_site_accessibility(self, url: str) -> bool:
        """Test if a website is accessible for scraping."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"🌐 Testing {url}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                content = response.text.lower()
                if "agent" in content or "realtor" in content:
                    print("   ✅ Accessible with agent content")
                    return True
                else:
                    print("   ⚠️ Accessible but no agent content")
                    return False
            elif response.status_code == 403:
                print("   ❌ Blocked (403 Forbidden)")
                return False
            else:
                print(f"   ❌ Not accessible (Status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def scrape_with_scrapegraph(self, url: str) -> List[Dict]:
        """Use ScrapeGraph AI to extract agent data."""
        try:
            print(f"🤖 Extracting data from: {url}")
            
            scraper = SmartScraperGraph(
                prompt=self.create_extraction_prompt(),
                source=url,
                config=self.config
            )
            
            result = scraper.run()
            
            # Handle different result formats
            if isinstance(result, dict):
                if 'content' in result:
                    return result['content'] if isinstance(result['content'], list) else [result['content']]
                else:
                    return [result]
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            print(f"❌ ScrapeGraph AI error: {e}")
            return []
    
    def scrape_state_license_database(self, state: str, search_term: str = "") -> List[Dict]:
        """Scrape state real estate license databases."""
        
        state_databases = {
            "texas": "https://www.trec.texas.gov/apps/license-holder-search",
            "california": "https://www2.dre.ca.gov/PublicASP/pplinfo.asp", 
            "florida": "https://www.myfloridalicense.com/CheckLicenseII/",
            "newyork": "https://appext20.dos.ny.gov/lcns_public/"
        }
        
        if state.lower() not in state_databases:
            print(f"❌ State '{state}' database not configured")
            return []
        
        url = state_databases[state.lower()]
        print(f"🏛️ Accessing {state.title()} license database: {url}")
        
        # Note: These require specific form submissions and would need
        # custom implementation for each state's interface
        print("⚠️ State databases require manual implementation due to complex forms")
        return []
    
    def try_alternative_sites(self, location: str = "") -> List[Dict]:
        """Try alternative real estate websites."""
        
        alternative_sites = [
            "https://www.northstarmls.com/",
            "https://www.mlspin.com/",
            "https://www.crmls.org/",
            "https://www.nwmls.com/",
            "https://www.ctmls.com/"
        ]
        
        all_agents = []
        
        for site in alternative_sites:
            print(f"\n🔍 Trying alternative site: {site}")
            
            if self.test_site_accessibility(site):
                agents = self.scrape_with_scrapegraph(site)
                if agents:
                    print(f"   ✅ Found {len(agents)} agents")
                    all_agents.extend(agents)
                else:
                    print("   ⚠️ No agents extracted")
            
            time.sleep(2)  # Be respectful
        
        return all_agents
    
    def scrape_agents(self, search_type: str = "alternative", **kwargs) -> List[RealEstateAgent]:
        """Main scraping method with multiple strategies."""
        
        print("🏠 Real Estate Agent Data Collection")
        print("=" * 40)
        print(f"🎯 Strategy: {search_type}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        raw_agents = []
        
        if search_type == "alternative":
            raw_agents = self.try_alternative_sites(kwargs.get('location', ''))
            
        elif search_type == "custom_url":
            url = kwargs.get('url')
            if url:
                if self.test_site_accessibility(url):
                    raw_agents = self.scrape_with_scrapegraph(url)
            else:
                print("❌ No URL provided for custom scraping")
                
        elif search_type == "state_license":
            state = kwargs.get('state', '')
            raw_agents = self.scrape_state_license_database(state)
        
        # Convert raw data to structured agent objects
        for agent_data in raw_agents:
            if isinstance(agent_data, dict):
                agent = RealEstateAgent(
                    name=agent_data.get('name'),
                    work=agent_data.get('work') or agent_data.get('office') or agent_data.get('brokerage'),
                    years_experience=agent_data.get('years_experience'),
                    recent_sales_count=agent_data.get('recent_sales_count') or agent_data.get('sales_last_12_months'),
                    recent_price_range=agent_data.get('recent_price_range') or agent_data.get('price_range'),
                    overall_rating=agent_data.get('overall_rating'),
                    review_count=agent_data.get('review_count'),
                    reviews=agent_data.get('reviews'),
                    active_listing_urls=agent_data.get('active_listing_urls') or agent_data.get('for_sale_urls'),
                    contact_number=agent_data.get('contact_number') or agent_data.get('phone'),
                    residence=agent_data.get('residence') or agent_data.get('location'),
                    email=agent_data.get('email'),
                    website=agent_data.get('website'),
                    specialties=agent_data.get('specialties'),
                    source_url=kwargs.get('url', 'alternative_sites'),
                    scraped_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                self.agents.append(agent)
        
        print(f"\n✅ Collection complete!")
        print(f"📊 Total agents found: {len(self.agents)}")
        
        return self.agents
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"real_estate_agents_{timestamp}.json"
        
        # Convert to dictionaries for JSON serialization
        agents_dict = [asdict(agent) for agent in self.agents]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(agents_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
    
    def display_summary(self):
        """Display a summary of collected data."""
        if not self.agents:
            print("📝 No agents collected yet.")
            return
        
        print(f"\n📋 Agent Collection Summary")
        print("=" * 30)
        print(f"Total Agents: {len(self.agents)}")
        
        # Count fields with data
        fields_with_data = {}
        for agent in self.agents:
            for field, value in asdict(agent).items():
                if value:
                    fields_with_data[field] = fields_with_data.get(field, 0) + 1
        
        print(f"\nData Completeness:")
        for field, count in sorted(fields_with_data.items()):
            percentage = (count / len(self.agents)) * 100
            print(f"  {field}: {count}/{len(self.agents)} ({percentage:.1f}%)")
        
        # Show sample agents
        print(f"\n📖 Sample Agents:")
        for i, agent in enumerate(self.agents[:3]):
            print(f"\n{i+1}. {agent.name or 'Unknown'}")
            if agent.work:
                print(f"   Company: {agent.work}")
            if agent.contact_number:
                print(f"   Phone: {agent.contact_number}")
            if agent.overall_rating:
                print(f"   Rating: {agent.overall_rating}/5 ({agent.review_count or 0} reviews)")

def main():
    """Main function demonstrating the complete solution."""
    
    print("🏠 Complete Real Estate Agent Data Collection Solution")
    print("=" * 56)
    print("Using ScrapeGraph AI + Gemini 1.5 Flash with fallback strategies")
    
    try:
        # Initialize scraper
        scraper = RealEstateAgentScraper()
        
        print(f"\n🎯 Available Collection Methods:")
        print("1. Alternative websites (less protected)")
        print("2. Custom URL (provide your own)")
        print("3. State license databases (manual implementation needed)")
        
        # Method 1: Try alternative sites
        print(f"\n🔍 Method 1: Trying alternative real estate sites...")
        agents = scraper.scrape_agents(search_type="alternative")
        
        if agents:
            scraper.display_summary()
            scraper.save_results()
        else:
            print("❌ No agents found from alternative sites")
            
            # Method 2: Try a custom URL (example)
            print(f"\n🔍 Method 2: Testing with a custom accessible URL...")
            test_url = "https://www.northstarmls.com/"  # From our earlier test
            agents = scraper.scrape_agents(search_type="custom_url", url=test_url)
            
            if agents:
                scraper.display_summary()
                scraper.save_results()
            else:
                print("❌ No agents found from custom URL either")
        
        # Provide recommendations
        print(f"\n🎯 RECOMMENDATIONS FOR YOUR PROJECT:")
        print("=" * 40)
        print("1. ✅ Use state license databases for verified data")
        print("2. ✅ Try LinkedIn Sales Navigator for professional data")
        print("3. ✅ Contact local real estate offices directly")
        print("4. ✅ Use regional MLS websites (less protected)")
        print("5. ✅ Consider paid APIs for large-scale collection")
        
        print(f"\n📚 DOCUMENTATION:")
        print("• ScrapeGraph AI successfully configured with Gemini 1.5 Flash")
        print("• Bot protection on major sites (Zillow, Realtor.com) prevents scraping")
        print("• Alternative approaches are more reliable and legal")
        print("• Your data model supports all requested fields")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        print(f"\n💡 Troubleshooting:")
        print("• Check your GOOGLE_API_KEY in .env file")
        print("• Ensure all packages are installed")
        print("• Try running individual components separately")

if __name__ == "__main__":
    main()

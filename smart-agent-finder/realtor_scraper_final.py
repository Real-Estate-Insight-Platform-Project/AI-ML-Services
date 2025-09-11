"""
Real Estate Agent Scraper - Final Implementation
==============================================

Complete solution using ScrapeGraph AI + Gemini 1.5 Flash with anti-bot protection handling.
"""

import os
import time
import random
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from scrapegraphai.graphs import SmartScraperGraph
from agent_schema import Agent
from utils.io import write_rows_csv

load_dotenv()

# Configuration
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# ScrapeGraph AI Configuration for Gemini 1.5 Flash
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4000
    },
    "verbose": False,
    "cache": True
}

class RealtorAgentScraper:
    """Main scraper class with anti-bot protection."""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
    def setup_browser(self):
        """Set up browser with anti-detection measures."""
        self.playwright = sync_playwright().start()
        
        # Launch with stealth settings
        self.browser = self.playwright.chromium.launch(
            headless=False,  # Visible browser helps avoid detection
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create context with realistic headers
        self.context = self.browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }
        )
        
        self.page = self.context.new_page()
        
        # Remove automation indicators
        self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
    
    def random_delay(self, min_sec=2, max_sec=5):
        """Random delay to mimic human behavior."""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def search_agents_by_zip(self, zipcode: str) -> List[str]:
        """
        Search for agents in a specific ZIP code and return profile URLs.
        """
        print(f"🔍 Searching agents in ZIP: {zipcode}")
        
        search_url = f"https://www.realtor.com/realestateagents/{zipcode}"
        
        try:
            # Load the search page
            print(f"📖 Loading: {search_url}")
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            self.random_delay(3, 7)
            
            # Check if blocked
            page_content = self.page.content().lower()
            if "blocked" in page_content or "reference id" in page_content:
                print("⚠️  Page appears to be blocked by anti-bot protection")
                print("🔧 Trying alternative approach...")
                
                # Try manual navigation
                input("Please manually navigate to the agents page and press Enter...")
            
            # Extract agent links using multiple strategies
            agent_links = self.extract_agent_links()
            
            if not agent_links:
                print("❌ No agent links found. Trying ScrapeGraph AI extraction...")
                agent_links = self.extract_links_with_ai()
            
            return agent_links
            
        except Exception as e:
            print(f"❌ Error searching agents: {e}")
            return []
    
    def extract_agent_links(self) -> List[str]:
        """Extract agent profile links using JavaScript."""
        try:
            links = self.page.evaluate("""
                () => {
                    const links = [];
                    const selectors = [
                        'a[href*="/realestateagents/"]',
                        'a[data-testid*="agent"]',
                        '.agent-card a',
                        '.RealtorCard a'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(element => {
                            const href = element.href;
                            if (href && href.includes('/realestateagents/') && !href.includes('/pg-')) {
                                links.push(href);
                            }
                        });
                    });
                    
                    return [...new Set(links)]; // Remove duplicates
                }
            """)
            
            print(f"✅ Found {len(links)} agent links via JavaScript")
            return links
            
        except Exception as e:
            print(f"❌ JavaScript extraction failed: {e}")
            return []
    
    def extract_links_with_ai(self) -> List[str]:
        """Use ScrapeGraph AI to extract agent links."""
        try:
            html_content = self.page.content()
            
            prompt = """
            Extract all real estate agent profile URLs from this webpage.
            Look for links that lead to individual agent profiles.
            Return as a JSON array of complete URLs only.
            Example: ["https://www.realtor.com/realestateagents/agent1", "https://www.realtor.com/realestateagents/agent2"]
            """
            
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=html_content,
                config=GRAPH_CONFIG
            )
            
            result = scraper.run()
            
            if isinstance(result, dict) and 'content' in result:
                links = result['content']
            elif isinstance(result, list):
                links = result
            else:
                links = []
            
            print(f"🤖 ScrapeGraph AI found {len(links)} links")
            return links
            
        except Exception as e:
            print(f"❌ AI extraction failed: {e}")
            return []
    
    def scrape_agent_profile(self, agent_url: str) -> Dict[str, Any]:
        """
        Scrape individual agent profile using ScrapeGraph AI.
        """
        print(f"👤 Scraping: {agent_url}")
        
        try:
            # Load agent profile
            self.page.goto(agent_url, wait_until="domcontentloaded", timeout=30000)
            self.random_delay(2, 4)
            
            # Click "Show more reviews" if available
            try:
                show_more_btn = self.page.locator("text=Show more reviews").first
                if show_more_btn.is_visible():
                    show_more_btn.click()
                    self.random_delay(1, 3)
            except:
                pass
            
            # Get HTML content
            html_content = self.page.content()
            
            # Extract data using ScrapeGraph AI
            agent_data = self.extract_agent_data_with_ai(html_content, agent_url)
            
            return agent_data
            
        except Exception as e:
            print(f"❌ Error scraping {agent_url}: {e}")
            return self.create_empty_agent_data(agent_url, str(e))
    
    def extract_agent_data_with_ai(self, html: str, profile_url: str) -> Dict[str, Any]:
        """Extract agent data using ScrapeGraph AI."""
        
        prompt = """
        Extract the following real estate agent information from this webpage:
        
        1. name: Agent's full name
        2. work_title: Job title or brokerage name
        3. years_experience: Years of experience (extract number if available)
        4. recent_sales_12mo: Number of recent sales (extract number)
        5. price_range: Price range they work with (e.g. "$200k - $500k")
        6. overall_rating: Star rating as a number (e.g. 4.5)
        7. review_count: Total number of reviews (extract number)
        8. reviews: Array of review objects with 'text' field
        9. active_listing_urls: Array of property listing URLs
        10. phones: Array of phone numbers
        11. location: Office address or service area
        
        Return as JSON format. Use null for missing fields, empty arrays for missing lists.
        """
        
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=html,
                config=GRAPH_CONFIG,
                schema=Agent.model_json_schema()
            )
            
            result = scraper.run()
            
            if isinstance(result, dict):
                data = result
            else:
                data = json.loads(result) if isinstance(result, str) else {}
            
            # Add metadata
            data["profile_url"] = profile_url
            data["scraped_at"] = pd.Timestamp.now().isoformat()
            
            print(f"✅ Extracted data for: {data.get('name', 'Unknown Agent')}")
            return data
            
        except Exception as e:
            print(f"❌ AI extraction failed: {e}")
            return self.create_empty_agent_data(profile_url, str(e))
    
    def create_empty_agent_data(self, profile_url: str, error: str = "") -> Dict[str, Any]:
        """Create empty agent data structure."""
        return {
            "name": None,
            "work_title": None,
            "years_experience": None,
            "recent_sales_12mo": None,
            "price_range": None,
            "overall_rating": None,
            "review_count": None,
            "reviews": [],
            "active_listing_urls": [],
            "phones": [],
            "location": None,
            "profile_url": profile_url,
            "scraped_at": pd.Timestamp.now().isoformat(),
            "extraction_error": error
        }
    
    def run_full_scrape(self, zip_codes: List[str], max_agents_per_zip: int = 10) -> List[Dict]:
        """
        Run full scraping process for multiple ZIP codes.
        """
        print(f"🚀 Starting full scrape for {len(zip_codes)} ZIP codes")
        
        all_agents = []
        
        try:
            self.setup_browser()
            
            for i, zipcode in enumerate(zip_codes, 1):
                print(f"\n📍 Processing ZIP {zipcode} ({i}/{len(zip_codes)})")
                
                # Get agent links
                agent_links = self.search_agents_by_zip(zipcode)
                
                if not agent_links:
                    print(f"⚠️  No agents found for ZIP {zipcode}")
                    continue
                
                # Limit agents per ZIP
                agent_links = agent_links[:max_agents_per_zip]
                
                # Scrape each agent
                for j, agent_url in enumerate(agent_links, 1):
                    print(f"  Agent {j}/{len(agent_links)}")
                    
                    agent_data = self.scrape_agent_profile(agent_url)
                    agent_data["zip"] = zipcode
                    
                    all_agents.append(agent_data)
                    
                    self.random_delay(2, 5)
                
                print(f"✅ Completed ZIP {zipcode}: {len(agent_links)} agents")
                self.random_delay(5, 10)  # Longer delay between ZIP codes
            
        finally:
            self.cleanup()
        
        return all_agents
    
    def cleanup(self):
        """Clean up browser resources."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

def main():
    """Main execution function."""
    
    # Test with a small set first
    test_zip_codes = ["82931", "10001", "90210"]  # Wyoming, NYC, Beverly Hills
    
    scraper = RealtorAgentScraper()
    
    try:
        agents = scraper.run_full_scrape(
            zip_codes=test_zip_codes[:1],  # Start with just one ZIP
            max_agents_per_zip=5
        )
        
        if agents:
            # Save to CSV
            output_file = "data/agents_final_scrapegraph.csv"
            write_rows_csv(agents, output_file)
            
            print(f"\n🎉 Scraping complete!")
            print(f"📊 Total agents scraped: {len(agents)}")
            print(f"💾 Data saved to: {output_file}")
            
            # Display sample data
            if agents:
                sample = agents[0]
                print(f"\n📋 Sample agent data:")
                for key, value in sample.items():
                    if key not in ['reviews', 'active_listing_urls']:
                        print(f"  {key}: {value}")
        else:
            print("❌ No agents were scraped")
            
    except KeyboardInterrupt:
        print("\n⏹️  Scraping stopped by user")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")

if __name__ == "__main__":
    main()

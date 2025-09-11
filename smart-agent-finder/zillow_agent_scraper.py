"""
Zillow Real Estate Agent Scraper using ScrapeGraph AI + Gemini 1.5 Flash
========================================================================

Complete solution to scrape real estate agent data from Zillow.com
Targets: Agent profiles, sales data, listings, reviews, and contact information.
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
from pydantic import BaseModel, Field
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

class ZillowAgent(BaseModel):
    """Pydantic model for Zillow agent data validation."""
    name: Optional[str] = Field(None, description="Agent's full name")
    brokerage: Optional[str] = Field(None, description="Brokerage/company name")
    years_experience: Optional[int] = Field(None, description="Years of experience")
    sales_last_12_months: Optional[int] = Field(None, description="Sales in last 12 months")
    total_sales: Optional[int] = Field(None, description="Total career sales")
    price_range: Optional[str] = Field(None, description="Price range (e.g., $18K-$914K)")
    average_price: Optional[str] = Field(None, description="Average sale price")
    overall_rating: Optional[float] = Field(None, description="Star rating")
    review_count: Optional[int] = Field(None, description="Number of reviews")
    reviews: List[Dict[str, Any]] = Field(default_factory=list, description="Customer reviews")
    for_sale_urls: List[str] = Field(default_factory=list, description="Current for-sale listing URLs")
    phones: List[str] = Field(default_factory=list, description="Contact phone numbers")
    email: Optional[str] = Field(None, description="Email address")
    location: Optional[str] = Field(None, description="Office location/service area")
    profile_url: Optional[str] = Field(None, description="Zillow profile URL")
    zip_code: Optional[str] = Field(None, description="Search ZIP code")
    scraped_at: Optional[str] = Field(None, description="Timestamp when scraped")

class ZillowAgentScraper:
    """Main scraper class for Zillow agent data."""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
    def setup_browser(self):
        """Set up browser with realistic settings."""
        self.playwright = sync_playwright().start()
        
        self.browser = self.playwright.chromium.launch(
            headless=False,  # Keep visible for debugging
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        self.context = self.browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
        Search for agents in a specific ZIP code on Zillow.
        Returns list of agent profile URLs.
        """
        print(f"🔍 Searching Zillow agents in ZIP: {zipcode}")
        
        search_url = f"https://www.zillow.com/professionals/real-estate-agent-reviews/{zipcode}/"
        
        try:
            print(f"📖 Loading: {search_url}")
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_load_state("networkidle", timeout=10000)
            self.random_delay(3, 6)
            
            # Extract agent profile links
            agent_links = self.extract_agent_links()
            
            # If no links found, try scrolling to load more
            if not agent_links:
                print("📜 Scrolling to load more agents...")
                self.scroll_and_load_more()
                agent_links = self.extract_agent_links()
            
            print(f"✅ Found {len(agent_links)} agent profiles")
            return agent_links
            
        except Exception as e:
            print(f"❌ Error searching agents: {e}")
            return []
    
    def scroll_and_load_more(self, max_scrolls=3):
        """Scroll page to load more agents."""
        for i in range(max_scrolls):
            try:
                # Scroll to bottom
                self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                self.random_delay(2, 4)
                
                # Look for "Load More" button
                load_more_btn = self.page.locator("text=Load more").first
                if load_more_btn.is_visible():
                    load_more_btn.click()
                    self.random_delay(2, 4)
                
            except Exception as e:
                print(f"⚠️  Scroll attempt {i+1} failed: {e}")
    
    def extract_agent_links(self) -> List[str]:
        """Extract agent profile links using JavaScript."""
        try:
            links = self.page.evaluate("""
                () => {
                    const links = [];
                    const selectors = [
                        'a[href*="/profile/"]',
                        'a[href*="/professionals/"]',
                        '[data-testid="agent-card"] a',
                        '.agent-card a'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(element => {
                            const href = element.href;
                            if (href && (href.includes('/profile/') || href.includes('/professionals/')) && !href.includes('/reviews/')) {
                                links.push(href);
                            }
                        });
                    });
                    
                    return [...new Set(links)]; // Remove duplicates
                }
            """)
            
            return links
            
        except Exception as e:
            print(f"❌ JavaScript extraction failed: {e}")
            return []
    
    def scrape_agent_profile(self, agent_url: str, zipcode: str) -> Dict[str, Any]:
        """
        Scrape individual agent profile using ScrapeGraph AI.
        """
        print(f"👤 Scraping: {agent_url}")
        
        try:
            # Load agent profile
            self.page.goto(agent_url, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_load_state("networkidle", timeout=10000)
            self.random_delay(2, 4)
            
            # Scroll to load all content
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.random_delay(1, 3)
            
            # Get HTML content
            html_content = self.page.content()
            
            # Extract data using ScrapeGraph AI
            agent_data = self.extract_agent_data_with_ai(html_content, agent_url, zipcode)
            
            return agent_data
            
        except Exception as e:
            print(f"❌ Error scraping {agent_url}: {e}")
            return self.create_empty_agent_data(agent_url, zipcode, str(e))
    
    def extract_agent_data_with_ai(self, html: str, profile_url: str, zipcode: str) -> Dict[str, Any]:
        """Extract agent data using ScrapeGraph AI."""
        
        prompt = """
        Extract the following real estate agent information from this Zillow profile page:
        
        REQUIRED FIELDS:
        1. name: Agent's full name
        2. brokerage: Brokerage/company name (e.g., "Keller Williams Realty")
        3. years_experience: Years of experience as number (e.g., 21)
        4. sales_last_12_months: Sales in last 12 months as number (e.g., 17)
        5. total_sales: Total career sales as number (e.g., 575)
        6. price_range: Price range (e.g., "$18K-$914K")
        7. average_price: Average sale price (e.g., "$309K")
        8. overall_rating: Star rating as decimal (e.g., 5.0)
        9. review_count: Number of reviews as number
        10. reviews: Array of review objects with 'text', 'rating', 'date' fields
        11. for_sale_urls: Array of current "For Sale" listing URLs (look for property listings)
        12. phones: Array of phone numbers
        13. email: Email address if visible
        14. location: Office location or service area
        
        EXTRACTION RULES:
        - Extract exact numbers without text (e.g., 17 not "17 sales")
        - For price_range, include the full range with $ signs
        - For for_sale_urls, get URLs to properties currently for sale by this agent
        - Use null for missing fields, empty arrays for missing lists
        - Be precise with numeric values
        
        Return as valid JSON matching this exact structure.
        """
        
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=html,
                config=GRAPH_CONFIG,
                schema=ZillowAgent.model_json_schema()
            )
            
            result = scraper.run()
            
            if isinstance(result, dict):
                data = result
            else:
                data = json.loads(result) if isinstance(result, str) else {}
            
            # Add metadata
            data["profile_url"] = profile_url
            data["zip_code"] = zipcode
            data["scraped_at"] = pd.Timestamp.now().isoformat()
            
            agent_name = data.get('name', 'Unknown Agent')
            print(f"✅ Extracted data for: {agent_name}")
            
            # Print key metrics
            sales_12mo = data.get('sales_last_12_months')
            total_sales = data.get('total_sales')
            experience = data.get('years_experience')
            
            if sales_12mo or total_sales or experience:
                print(f"   📊 {sales_12mo} sales (12mo) | {total_sales} total | {experience} years exp")
            
            return data
            
        except Exception as e:
            print(f"❌ AI extraction failed: {e}")
            return self.create_empty_agent_data(profile_url, zipcode, str(e))
    
    def create_empty_agent_data(self, profile_url: str, zipcode: str, error: str = "") -> Dict[str, Any]:
        """Create empty agent data structure."""
        return {
            "name": None,
            "brokerage": None,
            "years_experience": None,
            "sales_last_12_months": None,
            "total_sales": None,
            "price_range": None,
            "average_price": None,
            "overall_rating": None,
            "review_count": None,
            "reviews": [],
            "for_sale_urls": [],
            "phones": [],
            "email": None,
            "location": None,
            "profile_url": profile_url,
            "zip_code": zipcode,
            "scraped_at": pd.Timestamp.now().isoformat(),
            "extraction_error": error
        }
    
    def run_full_scrape(self, zip_codes: List[str], max_agents_per_zip: int = 20) -> List[Dict]:
        """
        Run full scraping process for multiple ZIP codes.
        """
        print(f"🚀 Starting Zillow agent scraping for {len(zip_codes)} ZIP codes")
        print(f"📊 Target: Max {max_agents_per_zip} agents per ZIP code")
        
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
                print(f"🎯 Processing {len(agent_links)} agents")
                
                # Scrape each agent
                for j, agent_url in enumerate(agent_links, 1):
                    print(f"  Agent {j}/{len(agent_links)}")
                    
                    agent_data = self.scrape_agent_profile(agent_url, zipcode)
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
    
    # Test ZIP codes - start with locations that likely have many agents
    test_zip_codes = [
        "35004",  # From your screenshot (Moody, AL)
        "10001",  # NYC
        "90210",  # Beverly Hills
        "33101",  # Miami
        "75201"   # Dallas
    ]
    
    scraper = ZillowAgentScraper()
    
    try:
        # Start with one ZIP code for testing
        agents = scraper.run_full_scrape(
            zip_codes=test_zip_codes[:1],  # Just first ZIP for testing
            max_agents_per_zip=10
        )
        
        if agents:
            # Save to CSV
            output_file = "data/zillow_agents.csv"
            write_rows_csv(agents, output_file)
            
            print(f"\n🎉 Zillow scraping complete!")
            print(f"📊 Total agents scraped: {len(agents)}")
            print(f"💾 Data saved to: {output_file}")
            
            # Display summary statistics
            df = pd.DataFrame(agents)
            
            print(f"\n📈 Summary Statistics:")
            print(f"   - Unique ZIP codes: {df['zip_code'].nunique()}")
            print(f"   - Agents with experience data: {df['years_experience'].notna().sum()}")
            print(f"   - Agents with sales data: {df['sales_last_12_months'].notna().sum()}")
            print(f"   - Total for-sale listings found: {sum(len(urls) for urls in df['for_sale_urls'] if urls)}")
            
            # Show top performers
            top_performers = df[df['sales_last_12_months'].notna()].nlargest(3, 'sales_last_12_months')
            if not top_performers.empty:
                print(f"\n🏆 Top Performers (Last 12 Months):")
                for _, agent in top_performers.iterrows():
                    print(f"   - {agent['name']}: {agent['sales_last_12_months']} sales")
            
            # Display sample data
            if agents:
                sample = agents[0]
                print(f"\n📋 Sample agent data:")
                for key, value in sample.items():
                    if key not in ['reviews', 'for_sale_urls'] and value is not None:
                        print(f"  {key}: {value}")
        else:
            print("❌ No agents were scraped")
            
    except KeyboardInterrupt:
        print("\n⏹️  Scraping stopped by user")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")

if __name__ == "__main__":
    main()

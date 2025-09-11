"""
Simple Real Estate Agent Scraper using ScrapeGraph AI
=====================================================

A simplified version for quick testing and demonstration.
This version focuses on a single ZIP code and provides detailed logging.
"""

import os
import time
import random
import json
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from scrapegraphai.graphs import SmartScraperGraph

# Load environment variables
load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# ScrapeGraph AI Configuration
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4000  # Fixed parameter name
    },
    "verbose": False,
    "cache": False,
    "headless": True
}

def simple_agent_search(zipcode: str = "82931"):
    """
    Simple function to test agent scraping for a single ZIP code.
    
    Args:
        zipcode: ZIP code to search (default: 82931 from your screenshots)
    """
    print(f"🔍 Searching for agents in ZIP code: {zipcode}")
    
    # Build search URL
    search_url = f"https://www.realtor.com/realestateagents/{zipcode}"
    print(f"📍 Search URL: {search_url}")
    
    # Simple extraction prompt
    listing_prompt = """
    Extract all real estate agent profile links from this page.
    Look for links that lead to individual agent profiles.
    Return as a JSON array of URLs.
    Example: ["https://www.realtor.com/realestateagents/agent1", "https://www.realtor.com/realestateagents/agent2"]
    """
    
    try:
        # Extract agent links using ScrapeGraph AI
        print("🤖 Using ScrapeGraph AI to find agent links...")
        
        link_scraper = SmartScraperGraph(
            prompt=listing_prompt,
            source=search_url,
            config=GRAPH_CONFIG
        )
        
        links_result = link_scraper.run()
        print(f"🔗 Links found: {links_result}")
        
        # Parse links
        if isinstance(links_result, dict):
            if 'content' in links_result:
                agent_links = links_result['content']
            else:
                agent_links = list(links_result.values())[0] if links_result else []
        elif isinstance(links_result, list):
            agent_links = links_result
        elif isinstance(links_result, str):
            try:
                agent_links = json.loads(links_result)
            except:
                agent_links = []
        else:
            agent_links = []
        
        if not agent_links or len(agent_links) == 0:
            print("⚠️  No agent links found, using sample URL for testing...")
            # Use a sample agent URL for testing
            first_agent_url = "https://www.realtor.com/realestateagents/simply-real-estate_5ff7e3ca1e5dd70100f54fe8"
        else:
            print(f"✅ Found {len(agent_links)} agent profiles")
            # Get first valid agent URL
            first_agent_url = None
            for link in agent_links:
                if isinstance(link, str) and link.startswith('http'):
                    first_agent_url = link
                    break
            
            if not first_agent_url:
                print("❌ No valid URLs found in results")
                return
        
        print(f"👤 Extracting data from: {first_agent_url}")
        
        # Detailed agent extraction prompt
        agent_prompt = """
        Extract complete real estate agent information from this profile page:
        
        Required fields:
        - name: Agent's full name
        - work_title: Job title or brokerage name  
        - years_experience: Years of experience
        - recent_sales_12mo: Recent sales count
        - price_range: Price range they work with
        - overall_rating: Star rating (numeric)
        - review_count: Number of reviews
        - reviews: Array of review text
        - phones: Contact phone numbers
        - location: Office address or service area
        
        Return as valid JSON with all available information.
        """
        
        agent_scraper = SmartScraperGraph(
            prompt=agent_prompt,
            source=first_agent_url,
            config=GRAPH_CONFIG
        )
        
        agent_data = agent_scraper.run()
        print(f"\n📋 Agent Data Extracted:")
        print(json.dumps(agent_data, indent=2, ensure_ascii=False))
        
        return agent_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_with_playwright():
    """Test using Playwright to load page first, then pass HTML to ScrapeGraph AI."""
    
    zipcode = "82931"
    search_url = f"https://www.realtor.com/realestateagents/{zipcode}"
    
    print(f"🌐 Loading page with Playwright: {search_url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to True for headless
        page = browser.new_page()
        
        try:
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle")
            
            # Get page HTML
            html_content = page.content()
            print(f"📄 Page loaded, HTML length: {len(html_content)} characters")
            
            # Use ScrapeGraph AI with HTML content
            link_prompt = """
            Find all real estate agent profile links on this page.
            Look for links that contain '/realestateagents/' and lead to individual agent profiles.
            Return as a simple JSON array of complete URLs.
            """
            
            link_scraper = SmartScraperGraph(
                prompt=link_prompt,
                source=html_content,  # Pass HTML instead of URL
                config=GRAPH_CONFIG
            )
            
            result = link_scraper.run()
            print(f"🔗 ScrapeGraph AI result: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    print("🚀 Starting Simple Agent Scraper Test")
    print("=" * 50)
    
    # Test 1: Direct URL scraping
    print("\n🧪 Test 1: Direct URL scraping with ScrapeGraph AI")
    simple_agent_search("82931")
    
    # Test 2: Playwright + ScrapeGraph AI
    print("\n🧪 Test 2: Playwright + ScrapeGraph AI")
    test_with_playwright()

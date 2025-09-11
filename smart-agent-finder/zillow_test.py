"""
Simple Zillow Agent Scraper Test
================================

Quick test to validate Zillow scraping with ScrapeGraph AI + Gemini 1.5 Flash.
"""

import os
import json
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from scrapegraphai.graphs import SmartScraperGraph

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# ScrapeGraph AI Configuration for Zillow
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4000
    },
    "verbose": True,
    "cache": False
}

def test_zillow_agent_search():
    """Test Zillow agent search and profile extraction."""
    
    # Using the ZIP code from your screenshot
    zipcode = "35004"  # Moody, AL
    search_url = f"https://www.zillow.com/professionals/real-estate-agent-reviews/{zipcode}/"
    
    print(f"🏠 Testing Zillow agent search for ZIP: {zipcode}")
    print(f"📍 URL: {search_url}")
    
    # Test 1: Extract agent links from search page
    link_prompt = """
    Extract all real estate agent profile links from this Zillow search page.
    Look for links that lead to individual agent profiles.
    Return as a JSON array of complete URLs.
    Focus on links containing '/profile/' or '/professionals/'.
    """
    
    try:
        print("\n🤖 Test 1: Extracting agent links with ScrapeGraph AI...")
        
        link_scraper = SmartScraperGraph(
            prompt=link_prompt,
            source=search_url,
            config=GRAPH_CONFIG
        )
        
        links_result = link_scraper.run()
        print(f"🔗 Links found: {links_result}")
        
        # Parse and validate links
        if isinstance(links_result, dict) and 'content' in links_result:
            agent_links = links_result['content']
        elif isinstance(links_result, list):
            agent_links = links_result
        else:
            agent_links = []
        
        if agent_links and len(agent_links) > 0:
            # Test with first agent profile
            first_agent_url = agent_links[0] if isinstance(agent_links, list) else agent_links
            print(f"✅ Found agents! Testing with: {first_agent_url}")
            
            # Test 2: Extract detailed agent data
            agent_prompt = """
            Extract comprehensive real estate agent data from this Zillow profile:
            
            - name: Agent's full name
            - brokerage: Company/brokerage name
            - years_experience: Years of experience (number only)
            - sales_last_12_months: Sales in last 12 months (number only)
            - total_sales: Total career sales (number only)
            - price_range: Price range (e.g., "$18K-$914K")
            - average_price: Average sale price
            - overall_rating: Star rating (number)
            - review_count: Number of reviews (number)
            - for_sale_urls: Current "For Sale" listing URLs
            - phones: Contact phone numbers
            - location: Office location
            
            Return as structured JSON with exact field names.
            """
            
            print(f"\n🤖 Test 2: Extracting agent profile data...")
            
            agent_scraper = SmartScraperGraph(
                prompt=agent_prompt,
                source=first_agent_url,
                config=GRAPH_CONFIG
            )
            
            agent_data = agent_scraper.run()
            
            print(f"\n📊 Agent Profile Data:")
            print(json.dumps(agent_data, indent=2, ensure_ascii=False))
            
            return agent_data
            
        else:
            print("❌ No agent links found")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def test_with_manual_url():
    """Test with a manually provided Zillow agent URL."""
    
    # Example agent URL (you can replace with actual URL from Zillow)
    agent_url = "https://www.zillow.com/profile/Jason-Secor"  # From your example
    
    print(f"\n🧪 Testing with manual agent URL: {agent_url}")
    
    agent_prompt = """
    Extract all available information from this Zillow real estate agent profile:
    
    SALES METRICS:
    - sales_last_12_months: Recent sales count
    - total_sales: Total career sales
    - years_experience: Years in business
    - price_range: Price range they handle
    - average_price: Average sale price
    
    PROFILE DATA:
    - name: Full name
    - brokerage: Company name
    - overall_rating: Star rating
    - review_count: Number of reviews
    - location: Service area/office
    
    CONTACT INFO:
    - phones: Phone numbers
    - email: Email address
    
    LISTINGS:
    - for_sale_urls: Current "For Sale" property URLs
    
    Return as JSON with exact field names above.
    """
    
    try:
        scraper = SmartScraperGraph(
            prompt=agent_prompt,
            source=agent_url,
            config=GRAPH_CONFIG
        )
        
        result = scraper.run()
        
        print(f"📋 Manual URL Test Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        print(f"❌ Manual URL test failed: {e}")
        return None

def test_with_playwright():
    """Test using Playwright to load Zillow page first."""
    
    zipcode = "35004"
    search_url = f"https://www.zillow.com/professionals/real-estate-agent-reviews/{zipcode}/"
    
    print(f"\n🌐 Testing with Playwright + ScrapeGraph AI")
    print(f"📖 Loading: {search_url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for debugging
        page = browser.new_page()
        
        try:
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle")
            
            # Get page content
            html_content = page.content()
            print(f"📄 Page loaded, HTML length: {len(html_content)} characters")
            
            # Check if page loaded successfully (look for Zillow content)
            if "zillow" in html_content.lower() and "agent" in html_content.lower():
                print("✅ Zillow page loaded successfully")
                
                # Extract agent data using ScrapeGraph AI
                prompt = """
                Find all real estate agents on this Zillow search page.
                For each agent, extract: name, rating, sales data, and profile URL.
                Return as JSON array of agent objects.
                """
                
                scraper = SmartScraperGraph(
                    prompt=prompt,
                    source=html_content,
                    config=GRAPH_CONFIG
                )
                
                result = scraper.run()
                print(f"🏠 Zillow Agents Found:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
            else:
                print("❌ Page doesn't appear to be Zillow content")
                print(f"Sample content: {html_content[:500]}...")
            
        except Exception as e:
            print(f"❌ Playwright test failed: {e}")
        finally:
            browser.close()

def main():
    """Run all Zillow tests."""
    print("🏠 Zillow Real Estate Agent Scraper Test Suite")
    print("=" * 60)
    
    tests = [
        ("Direct URL Scraping", test_zillow_agent_search),
        ("Manual URL Test", test_with_manual_url),
        ("Playwright + ScrapeGraph AI", test_with_playwright)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
        
        print("\nPress Enter to continue to next test...")
        input()

if __name__ == "__main__":
    main()

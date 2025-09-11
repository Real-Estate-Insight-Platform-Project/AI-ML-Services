"""
Working Real Estate Agent Scraper - Anti-Bot Protection Solution
==============================================================

This script bypasses anti-bot protection by using stealth techniques and manual intervention.
"""

import os
import time
import json
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from scrapegraphai.graphs import SmartScraperGraph

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# Fixed ScrapeGraph AI Configuration
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4000
    },
    "verbose": False,
    "cache": False
}

def bypass_antibot_and_scrape():
    """
    Method to bypass anti-bot protection and successfully scrape agent data.
    """
    print("🔒 Anti-Bot Bypass Method")
    print("=" * 40)
    
    with sync_playwright() as p:
        # Use stealth settings
        browser = p.chromium.launch(
            headless=False,  # Visible browser helps avoid detection
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = context.new_page()
        
        # Remove automation indicators
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        try:
            # Step 1: Go to main realtor.com page first
            print("🌐 Step 1: Loading main Realtor.com page...")
            page.goto("https://www.realtor.com", wait_until="domcontentloaded")
            time.sleep(3)
            
            # Step 2: Navigate to agent search
            print("🔍 Step 2: Navigating to agent search...")
            
            # Try to find and click "Find an Agent" link
            try:
                find_agent_link = page.locator("text=Find an Agent").first
                if find_agent_link.is_visible():
                    find_agent_link.click()
                    time.sleep(2)
            except:
                # Manual navigation
                page.goto("https://www.realtor.com/realestateagents", wait_until="domcontentloaded")
                time.sleep(2)
            
            # Step 3: Search for specific ZIP
            print("🏠 Step 3: Searching for ZIP code 82931...")
            
            # Try to find search input
            search_input = page.locator("input[placeholder*='ZIP'], input[name*='location'], input[type='search']").first
            if search_input.is_visible():
                search_input.fill("82931")
                time.sleep(1)
                
                # Press Enter or click search
                page.keyboard.press("Enter")
                time.sleep(3)
            else:
                # Direct navigation to ZIP page
                page.goto("https://www.realtor.com/realestateagents/82931", wait_until="domcontentloaded")
                time.sleep(3)
            
            # Step 4: Check if we got through
            page_content = page.content()
            
            if "Your request could not be processed" in page_content:
                print("🚫 Still blocked. Manual intervention required.")
                print("👤 Please manually navigate to the agents page and press Enter when ready...")
                input()
                page_content = page.content()
            
            print(f"📄 Page content length: {len(page_content)} characters")
            
            # Step 5: Extract agent links manually using JavaScript
            print("🔗 Step 5: Extracting agent links...")
            
            agent_links = page.evaluate("""
                () => {
                    const links = [];
                    
                    // Multiple selectors for agent links
                    const selectors = [
                        'a[href*="/realestateagents/"][href*="_"]',
                        'a[data-testid*="agent"]',
                        '.agent-card a',
                        '.RealtorCard a',
                        'a[href*="/realestate-agent/"]'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(link => {
                            const href = link.href;
                            if (href && href.includes('realestateagents') && !href.includes('/pg-')) {
                                links.push({
                                    url: href,
                                    text: link.textContent.trim().substring(0, 100)
                                });
                            }
                        });
                    });
                    
                    // Remove duplicates
                    const unique = [];
                    const seen = new Set();
                    
                    links.forEach(link => {
                        if (!seen.has(link.url)) {
                            seen.add(link.url);
                            unique.push(link);
                        }
                    });
                    
                    return unique;
                }
            """)
            
            print(f"🎯 Found {len(agent_links)} potential agent links:")
            for i, link in enumerate(agent_links[:5], 1):
                print(f"  {i}. {link['url']}")
                print(f"     Text: {link['text']}")
            
            if not agent_links:
                print("❌ No agent links found even with manual method")
                return None
            
            # Step 6: Scrape first agent profile
            first_agent = agent_links[0]
            agent_url = first_agent['url']
            
            print(f"\n👤 Step 6: Scraping agent profile: {agent_url}")
            
            page.goto(agent_url, wait_until="domcontentloaded")
            time.sleep(3)
            
            # Check for reviews section and click "Show more" if available
            try:
                show_more = page.locator("text=Show more reviews").first
                if show_more.is_visible():
                    show_more.click()
                    time.sleep(2)
            except:
                pass
            
            # Get the final HTML
            agent_html = page.content()
            
            # Step 7: Use ScrapeGraph AI to extract data
            print("🤖 Step 7: Extracting data with ScrapeGraph AI...")
            
            extraction_prompt = """
            Extract the following real estate agent information from this webpage:
            
            1. Agent name (usually in a large heading)
            2. Job title or brokerage name
            3. Years of experience 
            4. Number of recent sales
            5. Price range they work with
            6. Overall star rating
            7. Number of reviews
            8. Phone numbers
            9. Office location/address
            10. Any visible reviews
            
            Return as JSON format:
            {
                "name": "Agent Name",
                "brokerage": "Company Name", 
                "experience_years": 5,
                "recent_sales": 12,
                "price_range": "$200k - $500k",
                "rating": 4.8,
                "review_count": 25,
                "phone": "(555) 123-4567",
                "location": "City, State",
                "reviews": ["Great agent!", "Very helpful"]
            }
            
            Use null for missing information.
            """
            
            scraper = SmartScraperGraph(
                prompt=extraction_prompt,
                source=agent_html,
                config=GRAPH_CONFIG
            )
            
            result = scraper.run()
            
            print("\n📊 Extracted Agent Data:")
            print("=" * 30)
            
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"{key}: {value}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            
            return result
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            return None
            
        finally:
            browser.close()

def test_with_known_agent():
    """Test ScrapeGraph AI with a known working agent URL."""
    
    print("\n🧪 Testing with Known Agent URL")
    print("=" * 35)
    
    # Use a publicly available agent profile for testing
    test_url = "https://www.realtor.com/realestateagents/simply-real-estate_5ff7e3ca1e5dd70100f54fe8"
    
    extraction_prompt = """
    Extract real estate agent information from this page:
    
    - name: Agent's full name
    - brokerage: Company/brokerage name
    - phone: Contact phone number
    - rating: Star rating (number)
    - reviews: Number of reviews
    - location: Office location
    
    Return as JSON.
    """
    
    try:
        print(f"🔗 Testing URL: {test_url}")
        
        scraper = SmartScraperGraph(
            prompt=extraction_prompt,
            source=test_url,
            config=GRAPH_CONFIG
        )
        
        result = scraper.run()
        
        print("📋 Test Results:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def main():
    """Main function to run all tests."""
    
    print("🚀 Real Estate Agent Scraper - Working Solution")
    print("=" * 55)
    
    if not GOOGLE_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return
    
    print(f"✅ Google API Key loaded: {GOOGLE_KEY[:10]}...")
    
    # Test 1: Known agent URL
    test_with_known_agent()
    
    # Test 2: Anti-bot bypass method
    print("\n" + "="*55)
    user_input = input("🔄 Do you want to try the anti-bot bypass method? (y/n): ")
    
    if user_input.lower() == 'y':
        bypass_antibot_and_scrape()
    else:
        print("👋 Skipping anti-bot test. Run the script again if you want to try it.")

if __name__ == "__main__":
    main()

"""
Comprehensive Test Script for Real Estate Agent Scraping
========================================================

This script tests different approaches to verify what works best.
"""

import os
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from scrapegraphai.graphs import SmartScraperGraph
from bs4 import BeautifulSoup
import re

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

def test_url_patterns():
    """Test different URL patterns for agent listings."""
    
    zipcode = "82931"  # From your screenshots
    
    urls_to_test = [
        f"https://www.realtor.com/realestateagents/{zipcode}",
        f"https://www.realtor.com/realestateagents/{zipcode}/pg-1",
        f"https://www.realtor.com/realestateagents/{zipcode}/intent-buy",
        f"https://www.realtor.com/realestateagents/{zipcode}/intent-buy/pg-1",
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for debugging
        page = browser.new_page()
        
        for url in urls_to_test:
            print(f"\n🔍 Testing URL: {url}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_load_state("networkidle", timeout=10000)
                
                # Check page title and content
                title = page.title()
                print(f"📄 Page title: {title}")
                
                # Look for agent links manually
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Different patterns to search for
                patterns = [
                    "a[href*='/realestateagents/']",
                    "a[href*='/realestate-agent/']",
                    "[data-testid*='agent']",
                    ".agent-card",
                    ".RealtorCard"
                ]
                
                for pattern in patterns:
                    elements = soup.select(pattern)
                    if elements:
                        print(f"  ✅ Found {len(elements)} elements with pattern: {pattern}")
                        for i, elem in enumerate(elements[:3]):  # Show first 3
                            href = elem.get('href', '')
                            text = elem.get_text(strip=True)[:50]
                            print(f"    {i+1}. {href} | {text}")
                    else:
                        print(f"  ❌ No elements found with pattern: {pattern}")
                
                # Check for specific text
                page_text = soup.get_text()
                agent_count = page_text.count("agent")
                print(f"  📊 'agent' mentioned {agent_count} times on page")
                
                # Look for specific indicators
                if "agents found" in page_text.lower():
                    print(f"  ✅ Page shows agent search results")
                elif "no agents" in page_text.lower():
                    print(f"  ⚠️  Page indicates no agents found")
                else:
                    print(f"  ❓ Unclear if agents are shown")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error loading URL: {e}")
        
        browser.close()

def test_scrapegraph_with_known_agent():
    """Test ScrapeGraph AI with a known agent profile URL."""
    
    # This is a generic test - replace with actual agent URL from your search
    test_agent_url = "https://www.realtor.com/realestateagents/simply-real-estate_5ff7e3ca1e5dd70100f54fe8"
    
    config = {
        "llm": {
            "api_key": GOOGLE_KEY,
            "model": "google_genai/gemini-1.5-flash",
            "temperature": 0.1
        },
        "verbose": True
    }
    
    prompt = """
    Extract the following information from this real estate agent profile:
    
    1. Agent name
    2. Brokerage/company name
    3. Years of experience
    4. Contact phone number
    5. Office location
    6. Recent sales count
    7. Overall rating
    8. Number of reviews
    
    Return as JSON format.
    """
    
    try:
        print(f"🤖 Testing ScrapeGraph AI with agent URL...")
        
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=test_agent_url,
            config=config
        )
        
        result = scraper.run()
        print(f"📋 ScrapeGraph AI Result:")
        print(result)
        
    except Exception as e:
        print(f"❌ ScrapeGraph AI test failed: {e}")

def test_manual_agent_search():
    """Manual test to find and scrape agent data."""
    
    zipcode = "82931"
    search_url = f"https://www.realtor.com/realestateagents/{zipcode}"
    
    print(f"🔍 Manual agent search for ZIP {zipcode}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        
        try:
            print(f"📖 Loading: {search_url}")
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle")
            
            # Take screenshot for debugging
            page.screenshot(path="debug_search_page.png")
            print("📸 Screenshot saved as debug_search_page.png")
            
            # Wait for user to inspect
            print("🔍 Page loaded. Press Enter to continue with extraction...")
            input()
            
            # Get all links
            all_links = page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim().substring(0, 50)
                    })).filter(link => 
                        link.href.includes('realestateagents') || 
                        link.href.includes('agent')
                    );
                }
            """)
            
            print(f"🔗 Found {len(all_links)} potential agent links:")
            for i, link in enumerate(all_links[:10]):  # Show first 10
                print(f"  {i+1}. {link['href']}")
                print(f"     Text: {link['text']}")
            
            # Select first valid agent link
            agent_urls = [link['href'] for link in all_links 
                         if '/realestateagents/' in link['href'] and len(link['href'].split('/')) > 4]
            
            if agent_urls:
                first_agent = agent_urls[0]
                print(f"\n👤 Testing with first agent: {first_agent}")
                
                page.goto(first_agent, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle")
                
                # Take another screenshot
                page.screenshot(path="debug_agent_page.png")
                print("📸 Agent page screenshot saved as debug_agent_page.png")
                
                # Use ScrapeGraph AI on this specific page
                html_content = page.content()
                
                config = {
                    "llm": {
                        "api_key": GOOGLE_KEY,
                        "model": "google_genai/gemini-1.5-flash",
                        "temperature": 0.1
                    },
                    "verbose": True
                }
                
                prompt = """
                Extract agent information from this page:
                - Name
                - Job title/brokerage
                - Phone numbers
                - Years of experience
                - Recent sales
                - Rating and reviews
                - Location/address
                
                Return as structured JSON.
                """
                
                try:
                    scraper = SmartScraperGraph(
                        prompt=prompt,
                        source=html_content,
                        config=config
                    )
                    
                    result = scraper.run()
                    print(f"\n📋 Extracted Agent Data:")
                    print(result)
                    
                except Exception as e:
                    print(f"❌ Extraction failed: {e}")
            
            else:
                print("❌ No valid agent URLs found")
            
        except Exception as e:
            print(f"❌ Error in manual search: {e}")
        finally:
            browser.close()

def main():
    """Run all tests."""
    print("🧪 Real Estate Agent Scraping Test Suite")
    print("=" * 50)
    
    tests = [
        ("URL Pattern Testing", test_url_patterns),
        ("Manual Agent Search", test_manual_agent_search),
        # ("ScrapeGraph AI Test", test_scrapegraph_with_known_agent),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 30)
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("\nPress Enter to continue to next test...")
        input()

if __name__ == "__main__":
    main()

"""
Data-First Agent Scraper
=======================

This approach focuses on getting actual data by using multiple strategies.
"""

import os
import json
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# Optimized configuration
CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0,
        "max_tokens": 2000,
        "model_tokens": 1048576  # Gemini 1.5 Flash has 1M token context
    },
    "verbose": False
}

def test_sample_agent_data():
    """Test with a sample agent page that should work."""
    
    print("🧪 Testing with Sample Agent Data")
    print("=" * 40)
    
    # Sample HTML content of an agent page
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>John Smith - Real Estate Agent</title></head>
    <body>
        <h1>John Smith</h1>
        <p>Real Estate Agent at Keller Williams</p>
        <div class="stats">
            <span>15 years of experience</span>
            <span>23 sales in the last 12 months</span>
            <span>Price range: $250k - $750k</span>
        </div>
        <div class="rating">
            <span>4.8 stars</span>
            <span>47 reviews</span>
        </div>
        <div class="contact">
            <p>Phone: (555) 123-4567</p>
            <p>Office: 123 Main St, Denver, CO 80202</p>
        </div>
        <div class="reviews">
            <div class="review">
                <p>"John was amazing! Helped us find our dream home."</p>
                <span>5 stars - Posted 2024-01-15</span>
            </div>
            <div class="review">
                <p>"Professional and knowledgeable agent."</p>
                <span>4 stars - Posted 2024-01-10</span>
            </div>
        </div>
        <div class="listings">
            <h3>Active Listings</h3>
            <a href="/property/123">Beautiful 3BR Home - $450,000</a>
            <a href="/property/456">Modern Condo - $325,000</a>
        </div>
    </body>
    </html>
    """
    
    prompt = """
    Extract the following agent information:
    - name
    - work_title  
    - years_experience (number only)
    - recent_sales_12mo (number only)
    - price_range
    - overall_rating (number only) 
    - review_count (number only)
    - phones (array)
    - location
    - reviews (array of objects with text and stars)
    
    Return as clean JSON.
    """
    
    try:
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=sample_html,
            config=CONFIG
        )
        
        result = scraper.run()
        
        print("✅ Sample extraction successful!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        return None

def test_live_agent_url():
    """Test with a real agent URL."""
    
    print("\n🌐 Testing with Live Agent URL")
    print("=" * 40)
    
    # Try a few different agent URLs
    test_urls = [
        "https://www.realtor.com/realestateagents/simply-real-estate_5ff7e3ca1e5dd70100f54fe8",
        "https://www.realtor.com/realestateagents/jessica-lazier_56e1c99ce4b0c90c6c4e4b5b",
        "https://www.realtor.com/realestateagents/michelle-johnson_58b2f3d1e4b0c90c6c4e4b5c"
    ]
    
    prompt = """
    Extract agent data from this real estate agent profile:
    
    {
        "name": "agent full name",
        "brokerage": "company name",
        "phone": "phone number", 
        "rating": "star rating as number",
        "review_count": "number of reviews",
        "location": "office location",
        "experience": "years of experience"
    }
    
    Return valid JSON only.
    """
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n🔗 Test {i}: {url}")
        
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=CONFIG
            )
            
            result = scraper.run()
            
            if result and isinstance(result, dict) and any(result.values()):
                print("✅ Success! Got data:")
                print(json.dumps(result, indent=2))
                return result
            else:
                print("⚠️  No useful data extracted")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print("❌ All live URL tests failed")
    return None

def main():
    """Run comprehensive tests."""
    
    print("🚀 Data-First Agent Scraper Tests")
    print("=" * 50)
    
    if not GOOGLE_KEY:
        print("❌ GOOGLE_API_KEY not found in .env")
        return
    
    print(f"✅ API Key: {GOOGLE_KEY[:10]}...")
    
    # Test 1: Sample HTML
    sample_result = test_sample_agent_data()
    
    # Test 2: Live URLs  
    live_result = test_live_agent_url()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Sample HTML test: {'✅ PASS' if sample_result else '❌ FAIL'}")
    print(f"Live URL test: {'✅ PASS' if live_result else '❌ FAIL'}")
    
    if sample_result or live_result:
        print("\n🎉 ScrapeGraph AI is working! The issue is with Realtor.com blocking.")
        print("💡 Solutions:")
        print("  1. Use the working_scraper.py with anti-bot protection")
        print("  2. Try different real estate websites")
        print("  3. Use proxy servers or VPN")
        print("  4. Implement more sophisticated anti-detection")
    else:
        print("\n❌ ScrapeGraph AI configuration issue")
        print("🔧 Check your Google API key and model configuration")

if __name__ == "__main__":
    main()

"""
Debug Zillow Response Test
=========================

Let's see what Zillow actually returns to our requests.
"""

import requests
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def test_direct_request():
    """Test direct HTTP request to Zillow."""
    
    zipcode = "35004"
    url = f"https://www.zillow.com/professionals/real-estate-agent-reviews/{zipcode}/"
    
    print(f"🌐 Testing direct request to: {url}")
    
    # Headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Content Length: {len(response.text)} characters")
        
        # Save response for inspection
        with open("zillow_response.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        # Check for common blocking indicators
        content_lower = response.text.lower()
        
        if "access denied" in content_lower:
            print("❌ Access Denied - Zillow is blocking us")
        elif "robot" in content_lower or "bot" in content_lower:
            print("🤖 Bot detection triggered")
        elif "captcha" in content_lower:
            print("🔒 CAPTCHA challenge detected")
        elif "zillow" in content_lower and "agent" in content_lower:
            print("✅ Legitimate Zillow page received")
            
            # Parse with BeautifulSoup to check structure
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for agent-related elements
            agent_elements = soup.find_all('a', href=lambda x: x and 'profile' in x if x else False)
            print(f"🔗 Found {len(agent_elements)} profile links")
            
            if len(agent_elements) > 0:
                for i, elem in enumerate(agent_elements[:3]):  # Show first 3
                    print(f"   {i+1}. {elem.get('href')}")
            
            # Check for agent cards or listings
            agent_cards = soup.find_all(['div', 'section'], class_=lambda x: x and 'agent' in x.lower() if x else False)
            print(f"👥 Found {len(agent_cards)} agent-related elements")
            
        else:
            print("⚠️  Unknown response content")
            print(f"📝 First 500 characters:\n{response.text[:500]}")
        
        return response.text
        
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_alternative_zillow_urls():
    """Try different Zillow URL patterns."""
    
    base_urls = [
        "https://www.zillow.com/professionals/real-estate-agent-reviews/35004/",
        "https://www.zillow.com/homes/35004_rb/",
        "https://www.zillow.com/professionals/35004/",
        "https://www.zillow.com/agent-finder/35004/",
    ]
    
    for url in base_urls:
        print(f"\n🔍 Testing: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            print(f"   📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                content = response.text.lower()
                if "agent" in content and "zillow" in content:
                    print("   ✅ Contains agent data")
                else:
                    print("   ⚠️  No agent data detected")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def test_with_session():
    """Test with a persistent session."""
    
    print("\n🔐 Testing with persistent session...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # First visit main page
    try:
        main_page = session.get("https://www.zillow.com/", timeout=10)
        print(f"📄 Main page status: {main_page.status_code}")
        
        # Then try agent page
        agent_url = "https://www.zillow.com/professionals/real-estate-agent-reviews/35004/"
        agent_page = session.get(agent_url, timeout=10)
        print(f"👥 Agent page status: {agent_page.status_code}")
        
        if agent_page.status_code == 200:
            with open("zillow_session_response.html", "w", encoding="utf-8") as f:
                f.write(agent_page.text)
            print("✅ Session response saved to zillow_session_response.html")
        
    except Exception as e:
        print(f"❌ Session test failed: {e}")

def main():
    print("🏠 Zillow Response Debug Test")
    print("=" * 40)
    
    test_direct_request()
    test_alternative_zillow_urls()
    test_with_session()
    
    print("\n📁 Check the generated HTML files to see what Zillow actually returns.")

if __name__ == "__main__":
    main()

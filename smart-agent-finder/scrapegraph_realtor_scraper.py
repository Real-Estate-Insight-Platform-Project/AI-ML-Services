"""
Real Estate Agent Scraper using ScrapeGraph AI + Gemini 1.5 Flash
================================================================

This script implements a comprehensive solution to scrape real estate agent data
from Realtor.com using ScrapeGraph AI with Google's Gemini 1.5 Flash model.

Key Features:
- Multi-stage scraping: ZIP search → Agent listing → Individual profiles
- AI-powered data extraction using ScrapeGraph AI
- Handles dynamic content and "Show more reviews" interactions
- Structured data validation with Pydantic schemas
- Robust error handling and retry mechanisms
"""

import os
import time
import random
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from scrapegraphai.graphs import SmartScraperGraph
from agent_schema import Agent, Review
from utils.dom import collect_profile_links_from_listing, click_show_more_reviews
from utils.io import load_zip_series, write_rows_csv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APIKEY")
OUT_CSV = "data/agents_scrapegraph.csv"

# ScrapeGraph AI Configuration for Gemini 1.5 Flash
# Based on documentation: https://scrapegraph-ai.readthedocs.io/en/latest/scrapers/llm.html#gemini
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",  # Using google_genai provider
        "temperature": 0.1,
        "max_tokens": 4000
    },
    "verbose": True,
    "cache": True,
    "headless": True
}

# Detailed extraction prompt for agent data
AGENT_EXTRACTION_PROMPT = """
Extract all available real estate agent information from this webpage with precision:

REQUIRED FIELDS:
1. name: Agent's full name (usually in h1 or large header)
2. work_title: Job title/brokerage (second line under name, e.g., "Real Estate Agent at Keller Williams")
3. years_experience: Years of experience as text (look for "X years of experience")
4. recent_sales_12mo: Number of sales in past 12 months (look for "X sales" or "recent sales")
5. price_range: Price range they work with (look for "$XXX - $XXX" badges)
6. overall_rating: Star rating as number (e.g., 4.5, 4.9)
7. review_count: Total number of reviews (e.g., "21 reviews")
8. reviews: Array of review objects with 'text', 'date', and 'stars' fields
9. active_listing_urls: URLs from "Active Listings" or "Current Listings" section
10. phones: Phone numbers (mobile, office, etc.)
11. location: Office address or service area (city, state, ZIP)

EXTRACTION RULES:
- If any field is not found, use null or empty array as appropriate
- For numbers, extract only the numeric value
- For reviews, get ALL visible reviews after clicking "Show more"
- For phone numbers, extract in standard format
- For URLs, use full absolute URLs
- For location, get the complete address including ZIP code

Return data in valid JSON format matching the schema exactly.
"""

def random_sleep(min_sec: float = 0.8, max_sec: float = 1.8) -> None:
    """Random sleep to avoid being blocked."""
    time.sleep(random.uniform(min_sec, max_sec))

def build_listing_url(zipcode: str, page: int = 1) -> str:
    """Build URL for agent listing page by ZIP code."""
    return f"https://www.realtor.com/realestateagents/{zipcode}/pg-{page}"

def extract_agent_data_with_sgai(html: str, profile_url: str, zipcode: str) -> Dict[str, Any]:
    """
    Extract agent data using ScrapeGraph AI with Gemini 1.5 Flash.
    
    Args:
        html: Raw HTML content of the agent profile page
        profile_url: URL of the agent profile
        zipcode: ZIP code being processed
        
    Returns:
        Dictionary containing extracted agent data
    """
    try:
        print(f"    🤖 Extracting data with ScrapeGraph AI...")
        
        # Create SmartScraperGraph instance
        # Using HTML string as source to save bandwidth and improve reliability
        smart_scraper = SmartScraperGraph(
            prompt=AGENT_EXTRACTION_PROMPT,
            source=html,  # Pass HTML directly instead of URL
            config=GRAPH_CONFIG,
            schema=Agent.model_json_schema()  # Use Pydantic schema for validation
        )
        
        # Run the extraction
        result = smart_scraper.run()
        
        # Parse result
        if isinstance(result, dict):
            data = result
        elif isinstance(result, str):
            data = json.loads(result)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        # Add metadata
        data["profile_url"] = profile_url
        data["zip"] = zipcode
        data["scraped_at"] = pd.Timestamp.now().isoformat()
        
        print(f"    ✅ Successfully extracted data for: {data.get('name', 'Unknown')}")
        return data
        
    except Exception as e:
        print(f"    ❌ ScrapeGraph AI extraction failed: {e}")
        # Return minimal data structure
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
            "zip": zipcode,
            "scraped_at": pd.Timestamp.now().isoformat(),
            "extraction_error": str(e)
        }

def search_agents_by_zip(zip_codes: List[str], 
                        pages_per_zip: int = 2,
                        max_agents_per_zip: int = 20,
                        output_file: str = OUT_CSV) -> None:
    """
    Main scraping function to search agents by ZIP codes.
    
    Args:
        zip_codes: List of ZIP codes to search
        pages_per_zip: Number of listing pages to process per ZIP
        max_agents_per_zip: Maximum agents to scrape per ZIP
        output_file: Output CSV file path
    """
    print(f"🚀 Starting agent scraping with ScrapeGraph AI + Gemini 1.5 Flash")
    print(f"📊 Target: {len(zip_codes)} ZIP codes, {pages_per_zip} pages each, max {max_agents_per_zip} agents per ZIP")
    
    all_agents = []
    total_scraped = 0
    
    with sync_playwright() as playwright:
        # Launch browser with realistic settings
        browser = playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = context.new_page()
        
        # Process each ZIP code
        for zip_idx, zipcode in enumerate(zip_codes, 1):
            print(f"\n📍 Processing ZIP {zipcode} ({zip_idx}/{len(zip_codes)})")
            
            agents_found_this_zip = 0
            
            # Process multiple pages for this ZIP
            for page_num in range(1, pages_per_zip + 1):
                if agents_found_this_zip >= max_agents_per_zip:
                    print(f"   ⚡ Reached max agents limit for ZIP {zipcode}")
                    break
                
                listing_url = build_listing_url(zipcode, page_num)
                print(f"  📄 Page {page_num}: {listing_url}")
                
                try:
                    # Load listing page
                    page.goto(listing_url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_load_state("networkidle", timeout=10000)
                    random_sleep(1.0, 2.0)
                    
                    # Extract agent profile links
                    page_html = page.content()
                    agent_links = collect_profile_links_from_listing(page_html)
                    
                    if not agent_links:
                        print(f"    ⚠️  No agent profiles found on page {page_num}")
                        continue
                    
                    print(f"    🔗 Found {len(agent_links)} agent profiles")
                    
                    # Process each agent profile
                    for link_idx, agent_url in enumerate(agent_links):
                        if agents_found_this_zip >= max_agents_per_zip:
                            break
                        
                        print(f"    👤 Agent {link_idx + 1}/{len(agent_links)}: {agent_url}")
                        
                        try:
                            # Load agent profile page
                            page.goto(agent_url, wait_until="domcontentloaded", timeout=30000)
                            page.wait_for_load_state("networkidle", timeout=10000)
                            
                            # Click "Show more reviews" to load all reviews
                            click_show_more_reviews(page, max_clicks=5)
                            
                            # Get final page HTML
                            agent_html = page.content()
                            
                            # Extract agent data using ScrapeGraph AI
                            agent_data = extract_agent_data_with_sgai(agent_html, agent_url, zipcode)
                            
                            # Validate with Pydantic schema
                            try:
                                validated_agent = Agent(**agent_data)
                                all_agents.append(validated_agent.model_dump())
                                agents_found_this_zip += 1
                                total_scraped += 1
                                print(f"    ✅ Successfully scraped agent #{total_scraped}")
                            except Exception as validation_error:
                                print(f"    ⚠️  Validation failed: {validation_error}")
                                # Still save the raw data
                                all_agents.append(agent_data)
                                agents_found_this_zip += 1
                                total_scraped += 1
                            
                            random_sleep(1.5, 3.0)
                            
                        except PlaywrightTimeout:
                            print(f"    ⏰ Timeout loading agent page: {agent_url}")
                            continue
                        except Exception as e:
                            print(f"    ❌ Error processing agent {agent_url}: {e}")
                            continue
                    
                    random_sleep(2.0, 4.0)
                    
                except PlaywrightTimeout:
                    print(f"    ⏰ Timeout loading listing page {page_num}")
                    continue
                except Exception as e:
                    print(f"    ❌ Error processing listing page {page_num}: {e}")
                    continue
            
            print(f"  📊 ZIP {zipcode} complete: {agents_found_this_zip} agents scraped")
            random_sleep(3.0, 6.0)  # Longer pause between ZIP codes
        
        browser.close()
    
    # Save results
    if all_agents:
        write_rows_csv(all_agents, output_file)
        print(f"\n🎉 Scraping complete!")
        print(f"📈 Total agents scraped: {total_scraped}")
        print(f"💾 Data saved to: {output_file}")
        
        # Print summary statistics
        df = pd.DataFrame(all_agents)
        print(f"📊 Summary:")
        print(f"   - Unique ZIP codes: {df['zip'].nunique()}")
        print(f"   - Agents with ratings: {df['overall_rating'].notna().sum()}")
        print(f"   - Agents with reviews: {df['review_count'].notna().sum()}")
        print(f"   - Average rating: {df['overall_rating'].mean():.2f}" if df['overall_rating'].notna().any() else "   - No ratings found")
    else:
        print(f"\n⚠️  No agents were successfully scraped")

def main():
    """Main execution function."""
    # Load ZIP codes from CSV
    zip_csv_path = "data/US zip-state-city.csv"
    zip_column = "ZIP Code"
    
    try:
        zip_codes = list(load_zip_series(zip_csv_path, zip_column))
        print(f"📋 Loaded {len(zip_codes)} ZIP codes from {zip_csv_path}")
        
        # For testing, limit to first few ZIP codes
        test_zip_codes = zip_codes[:3]  # Change this for full run
        
        # Run the scraper
        search_agents_by_zip(
            zip_codes=test_zip_codes,
            pages_per_zip=2,
            max_agents_per_zip=10,  # Start small for testing
            output_file="data/agents_scrapegraph_test.csv"
        )
        
    except FileNotFoundError:
        print(f"❌ ZIP codes file not found: {zip_csv_path}")
        print("📝 Please ensure the ZIP codes CSV file exists")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")

if __name__ == "__main__":
    main()

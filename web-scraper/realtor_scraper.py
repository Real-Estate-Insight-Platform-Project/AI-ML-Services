import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urlencode, quote
import pandas as pd
from typing import Dict, List, Optional
import re
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """Data class to store user preferences for property search"""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_beds: Optional[int] = None
    max_beds: Optional[int] = None
    min_baths: Optional[int] = None
    max_baths: Optional[int] = None
    # single-family-home, condo, townhome, etc.
    property_type: str = "single-family-home"
    sort_by: str = "newest"  # newest, price-low-to-high, price-high-to-low


class RealtorScraper:
    """
    Web scraper for Realtor.com property listings with enhanced anti-detection measures
    """

    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://www.realtor.com"
        # List of user agents to rotate through
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        self.session.headers.update(self.headers)
        self.request_count = 0

    def rotate_user_agent(self):
        """Rotate user agent to avoid detection"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents)
        })

    def build_search_url(self, preferences: UserPreferences) -> str:
        """
        Build search URL based on user preferences using Realtor.com's actual URL structure
        """
        # Base location part
        if preferences.city and preferences.state:
            location = f"{preferences.city}_{preferences.state}"
        elif preferences.zip_code:
            location = preferences.zip_code
        else:
            raise ValueError("Either city/state or zip_code must be provided")

        # Start building URL path
        url_path = f"/realestateandhomes-search/{location}"

        # Add property type
        if preferences.property_type != "single-family-home":
            url_path += f"/type-{preferences.property_type}"

        # Add beds filter
        if preferences.min_beds and preferences.max_beds:
            url_path += f"/beds-{preferences.min_beds}-{preferences.max_beds}"
        elif preferences.min_beds:
            url_path += f"/beds-{preferences.min_beds}"
        elif preferences.max_beds:
            url_path += f"/beds-na-{preferences.max_beds}"

        # Add baths filter
        if preferences.min_baths and preferences.max_baths:
            url_path += f"/baths-{preferences.min_baths}-{preferences.max_baths}"
        elif preferences.min_baths:
            url_path += f"/baths-{preferences.min_baths}"
        elif preferences.max_baths:
            url_path += f"/baths-na-{preferences.max_baths}"

        # Add price filter
        if preferences.min_price and preferences.max_price:
            url_path += f"/price-{preferences.min_price}-{preferences.max_price}"
        elif preferences.min_price:
            url_path += f"/price-{preferences.min_price}"
        elif preferences.max_price:
            url_path += f"/price-na-{preferences.max_price}"

        # Add sorting
        sort_mapping = {
            "newest": "1",
            "price-low-to-high": "6",
            "price-high-to-low": "7"
        }
        sort_code = sort_mapping.get(preferences.sort_by, "1")
        url_path += f"/sby-{sort_code}"

        search_url = self.base_url + url_path
        logger.info(f"Built search URL: {search_url}")
        return search_url

    def get_page_content(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Fetch page content with retry mechanism and enhanced anti-detection measures
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being blocked
                delay = random.uniform(3, 8)  # Increased delay
                logger.info(f"Waiting {delay:.2f} seconds before request...")
                time.sleep(delay)

                # Rotate user agent
                self.rotate_user_agent()

                # Add referer header if not first request
                if self.request_count > 0:
                    self.session.headers.update({'Referer': self.base_url})

                response = self.session.get(url, timeout=15)
                response.raise_for_status()

                # Check if we got a valid HTML response
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    logger.warning(f"Received non-HTML response from {url}")
                    return None

                # Check for captcha or blocking in response text
                if any(blocked_indicator in response.text for blocked_indicator in
                       ['captcha', 'CAPTCHA', 'Access Denied', 'blocked', 'Too Many Requests']):
                    logger.warning(
                        f"Page appears to be blocked or showing captcha: {url}")
                    return None

                soup = BeautifulSoup(response.content, 'html.parser')

                # Check if the page contains property listings
                if not self.contains_properties(soup):
                    logger.warning(
                        f"Page doesn't appear to contain property listings: {url}")
                    return None

                self.request_count += 1
                return soup

            except requests.RequestException as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for URL {url}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to fetch {url} after {max_retries} attempts")
                    return None
                # Exponential backoff for retries
                time.sleep(random.uniform(5, 10) * (attempt + 1))

        return None

    def contains_properties(self, soup: BeautifulSoup) -> bool:
        """
        Check if the page contains property listings
        """
        # Check for various indicators of property listings
        indicators = [
            soup.find('div', {'data-testid': 'map-result-card'}),
            soup.find('div', class_=re.compile('property-card', re.I)),
            soup.find('div', class_=re.compile('listing', re.I)),
            soup.find('span', class_=re.compile('price', re.I)),
        ]
        return any(indicators)

    def extract_property_data(self, property_card) -> Optional[Dict]:
        """
        Extract property data from a property card element
        """
        try:
            property_data = {}

            # Extract price
            price_elem = property_card.find(
                ['span', 'div'], {'data-testid': 'card-price'})
            if not price_elem:
                price_elem = property_card.find(
                    ['span', 'div'], class_=re.compile('price'))

            if price_elem:
                price_text = price_elem.get_text(strip=True)
                # Extract numeric price
                price_match = re.search(
                    r'\$?([\d,]+)', price_text.replace(',', ''))
                if price_match:
                    property_data['price'] = int(
                        price_match.group(1).replace(',', ''))
                property_data['price_text'] = price_text

            # Extract address
            address_elem = property_card.find(
                ['div'], {'data-testid': 'card-address'})
            if not address_elem:
                address_elem = property_card.find(
                    ['div', 'span'], class_=re.compile('address'))

            if address_elem:
                property_data['address'] = address_elem.get_text(strip=True)

            # Extract beds, baths, sqft
            bed_bath_elem = property_card.find(
                ['ul', 'div'], {'data-testid': 'card-meta'})
            if not bed_bath_elem:
                bed_bath_elem = property_card.find(
                    ['ul', 'div'], class_=re.compile('meta'))

            if bed_bath_elem:
                meta_text = bed_bath_elem.get_text(strip=True)

                # Extract beds
                beds_match = re.search(
                    r'(\d+)\s*bed', meta_text, re.IGNORECASE)
                if beds_match:
                    property_data['beds'] = int(beds_match.group(1))

                # Extract baths
                baths_match = re.search(
                    r'(\d+(?:\.\d+)?)\s*bath', meta_text, re.IGNORECASE)
                if baths_match:
                    property_data['baths'] = float(baths_match.group(1))

                # Extract square footage
                sqft_match = re.search(
                    r'([\d,]+)\s*sq\s*ft', meta_text, re.IGNORECASE)
                if sqft_match:
                    property_data['sqft'] = int(
                        sqft_match.group(1).replace(',', ''))

            # Extract property link
            link_elem = property_card.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if href.startswith('/'):
                    href = self.base_url + href
                property_data['url'] = href

            # Extract property type
            type_elem = property_card.find(['span'], class_=re.compile('type'))
            if type_elem:
                property_data['property_type'] = type_elem.get_text(strip=True)

            # Extract listing date/status
            status_elem = property_card.find(
                ['div', 'span'], class_=re.compile('status'))
            if status_elem:
                property_data['status'] = status_elem.get_text(strip=True)

            # Extract broker/agent info if available
            broker_elem = property_card.find(
                ['div', 'span'], class_=re.compile('broker'))
            if broker_elem:
                property_data['broker'] = broker_elem.get_text(strip=True)

            # Only return if we have essential data
            if 'price' in property_data and 'address' in property_data:
                return property_data
            else:
                return None

        except Exception as e:
            logger.warning(f"Error extracting property data: {str(e)}")
            return None

    def scrape_properties(self, preferences: UserPreferences, max_pages: int = 3) -> List[Dict]:
        """
        Scrape properties based on user preferences with enhanced anti-detection
        """
        all_properties = []

        try:
            base_url = self.build_search_url(preferences)

            for page in range(1, max_pages + 1):
                logger.info(f"Scraping page {page}...")

                # Build page URL
                if page == 1:
                    page_url = base_url
                else:
                    # Add pagination parameter
                    page_url = base_url + f"/pg-{page}"

                soup = self.get_page_content(page_url)
                if not soup:
                    logger.warning(f"Failed to get content for page {page}")
                    # If we can't get the first page, break entirely
                    if page == 1:
                        break
                    continue

                # Find property cards with various possible selectors
                property_cards = []

                # Try different selectors that Realtor.com might use
                selectors = [
                    '[data-testid="property-card"]',
                    '.card-content',
                    '.property-card',
                    '[data-testid="card-container"]',
                    '.js-srp-listing-photos',
                    '.srp-item'
                ]

                for selector in selectors:
                    cards = soup.select(selector)
                    if cards:
                        property_cards = cards
                        logger.info(
                            f"Found {len(cards)} properties using selector: {selector}")
                        break

                if not property_cards:
                    # Fallback: look for any div that contains price information
                    all_divs = soup.find_all('div')
                    property_cards = [div for div in all_divs
                                      if div.find(text=re.compile(r'\$[\d,]+')) and
                                      div.find(text=re.compile(r'bed|bath', re.IGNORECASE))]
                    logger.info(
                        f"Fallback method found {len(property_cards)} potential property cards")

                if not property_cards:
                    logger.warning(f"No property cards found on page {page}")
                    break

                # Extract data from each property card
                page_properties = []
                for card in property_cards:
                    property_data = self.extract_property_data(card)
                    if property_data:
                        property_data['page'] = page
                        page_properties.append(property_data)

                logger.info(
                    f"Extracted {len(page_properties)} properties from page {page}")
                all_properties.extend(page_properties)

                # If we got very few properties, might have hit the end
                if len(page_properties) < 5:
                    logger.info(
                        "Few properties found, might be at end of results")
                    break

            logger.info(f"Total properties scraped: {len(all_properties)}")
            return all_properties

        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return all_properties

    def save_to_csv(self, properties: List[Dict], filename: str = "scraped_properties.csv"):
        """
        Save scraped properties to CSV file
        """
        if not properties:
            logger.warning("No properties to save")
            return

        df = pd.DataFrame(properties)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(properties)} properties to {filename}")

    def save_to_json(self, properties: List[Dict], filename: str = "scraped_properties.json"):
        """
        Save scraped properties to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=2, default=str)
        logger.info(f"Saved {len(properties)} properties to {filename}")


# Alternative approach using a different method if the main one fails
def try_alternative_scraping_method():
    """
    Alternative approach if the main scraping method fails due to blocking
    """
    logger.info("Trying alternative scraping method...")

    # You might try using a different user agent, longer delays, or
    # even a different library like selenium with a real browser

    # This is a placeholder for an alternative approach
    return []


def main():
    """
    Example usage of the RealtorScraper with enhanced anti-detection
    """
    scraper = RealtorScraper()

    # Define user preferences
    preferences = UserPreferences(
        city="Miami",
        state="FL",
        min_price=200000,
        max_price=500000,
        min_beds=2,
        max_beds=4,
        min_baths=2,
        property_type="single-family-home",
        sort_by="price-low-to-high"
    )

    print("Starting property scraping with enhanced anti-detection...")
    properties = scraper.scrape_properties(preferences, max_pages=2)

    if not properties:
        print("Main scraping method failed, trying alternative...")
        properties = try_alternative_scraping_method()

    if properties:
        print(f"Found {len(properties)} properties")

        # Display first few properties
        for i, prop in enumerate(properties[:5]):
            print(f"\n--- Property {i+1} ---")
            for key, value in prop.items():
                print(f"{key}: {value}")

        # Save results
        scraper.save_to_csv(properties, "miami_properties.csv")
        scraper.save_to_json(properties, "miami_properties.json")
    else:
        print("No properties found after trying all methods")
        print("This might be due to:")
        print("1. Realtor.com blocking scraping attempts")
        print("2. Changes to the website structure")
        print("3. Network issues")
        print("Consider using a proxy service or official API if available")


if __name__ == "__main__":
    main()

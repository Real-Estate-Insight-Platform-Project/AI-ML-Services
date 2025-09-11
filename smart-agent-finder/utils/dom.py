import re
from bs4 import BeautifulSoup
from playwright.sync_api import Page

# Updated regex patterns to handle different agent URL formats
PROFILE_RE = re.compile(r"^/realestateagents/[^/]+$", re.I)
PROFILE_ID_RE = re.compile(r"^/realestateagents/[0-9a-f-]+$", re.I)

def collect_profile_links_from_listing(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    
    # Multiple selection strategies for different page layouts
    selectors = [
        "a[href*='/realestateagents/']",
        "a[data-testid*='agent-card']", 
        "a[href*='/realestate-agent/']",
        ".agent-card a",
        "[data-testid='agent-card'] a",
        ".RealtorCard a"
    ]
    
    for selector in selectors:
        elements = soup.select(selector)
        for a in elements:
            href = a.get("href", "")
            if href and ("/realestateagents/" in href or "/realestate-agent/" in href):
                # Clean and validate the href
                href = href.strip()
                if href.startswith("http"):
                    urls.append(href)
                elif href.startswith("/"):
                    urls.append("https://www.realtor.com" + href)
    
    # Alternative: Look for agent names that link to profiles
    agent_links = soup.find_all("a", string=True)
    for link in agent_links:
        href = link.get("href", "")
        if href and "/realestateagents/" in href:
            if href.startswith("/"):
                href = "https://www.realtor.com" + href
            urls.append(href)
    
    # Deduplicate while preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen and "realestateagents" in u:
            seen.add(u)
            out.append(u)
    
    print(f"Found {len(out)} unique agent profile URLs")
    return out


def click_show_more_reviews(page: Page, max_clicks: int = 8):
    # Try the “Ratings and reviews” tab and then keep clicking “Show more reviews”
    try:
        page.get_by_text("Ratings and reviews", exact=False).click(timeout=2000)
        page.wait_for_load_state("networkidle")
    except Exception:
        pass
    for _ in range(max_clicks):
        try:
            locator = page.get_by_text("Show more reviews", exact=False)
            if locator.is_visible():
                locator.click(timeout=1500)
                page.wait_for_load_state("networkidle")
            else:
                break
        except Exception:
            break

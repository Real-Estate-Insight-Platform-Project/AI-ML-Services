import os, time, random, json
from dotenv import load_dotenv
from typing import Optional
from playwright.sync_api import sync_playwright
from scrapegraphai.graphs import SmartScraperGraph
from agent_schema import Agent, Review
from utils.dom import collect_profile_links_from_listing, click_show_more_reviews
from utils.io import load_zip_series, write_rows_csv

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APIKEY")
OUT_CSV = "data/agents.csv"

# --- ScrapeGraphAI + Gemini 1.5 Flash config ---
# See examples: specify provider prefix with the model id (google_genai or google_vertexai). :contentReference[oaicite:8]{index=8}
GRAPH_CONFIG = {
    "llm": {
        "api_key": GOOGLE_KEY,
        "model": "google_genai/gemini-1.5-flash",
        "temperature": 0
    },
    "verbose": False,
    "cache": True
}

PROMPT = (
    "From this page, extract the agent data as follows:\n"
    "- name\n"
    "- work_title (the second line under the name, e.g., brokerage/role)\n"
    "- years_experience (text if shown)\n"
    "- recent_sales_12mo (sales in the last 12 months if shown)\n"
    "- price_range (the ‘price range’ badge on the page)\n"
    "- overall_rating (numeric if visible)\n"
    "- review_count (numeric if visible)\n"
    "- reviews: an array of objects with only 'text' (click and load more before extracting)\n"
    "- active_listing_urls: array of URLs in 'Active Listings'\n"
    "- phones: any phone numbers on the page\n"
    "- location: mailing or office address city/state/ZIP block\n"
    "Return STRICTLY in the schema. If a field is missing on the page, use a sensible empty value."
)

def rand_sleep(a=0.8, b=1.8): time.sleep(random.uniform(a, b))

def listing_url(zipcode: str, page: int) -> str:
    return f"https://www.realtor.com/realestateagents/{zipcode}/pg-{page}"

def extract_with_sgai(html: str, profile_url: str, state: Optional[str], zipcode: str) -> dict:
    # ScrapeGraphAI accepts URL OR HTML string as source; passing HTML saves bandwidth. :contentReference[oaicite:9]{index=9}
    smart = SmartScraperGraph(
        prompt=PROMPT,
        source=html,
        config=GRAPH_CONFIG,
        schema=Agent.model_json_schema()  # Pydantic JSON schema
    )
    result = smart.run()
    data = result if isinstance(result, dict) else json.loads(result)
    data["profile_url"] = profile_url
    data["state"] = state or ""
    data["zip"] = zipcode
    return data

def run(zip_csv="data/US zip-state-city.csv", zip_col="ZIP Code",
        pages_per_zip=2, limit_profiles_per_zip=20, out_csv=OUT_CSV):
    rows = []
    zips = list(load_zip_series(zip_csv, zip_col))  # zfill to 5 already
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent="Mozilla/5.0 (compatible; SmartAgentFinderBot/1.0; +https://example.org/bot-info)"
        )
        page = ctx.new_page()

        for zipcode in zips:
            scraped_this_zip = 0
            for p in range(1, pages_per_zip + 1):
                url = listing_url(zipcode, p)
                print(f"[{zipcode}] listing {p}: {url}")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_load_state("networkidle")
                except Exception as e:
                    print(f"  listing load failed: {e}")
                    break

                links = collect_profile_links_from_listing(page.content())
                if not links:
                    print("  no profile links found on this page")
                    continue

                for prof in links:
                    if scraped_this_zip >= limit_profiles_per_zip:
                        break
                    try:
                        page.goto(prof, wait_until="domcontentloaded", timeout=60_000)
                        page.wait_for_load_state("networkidle")
                        click_show_more_reviews(page, max_clicks=8)
                        html = page.content()
                        data = extract_with_sgai(html, prof, state=None, zipcode=zipcode)
                        # Validate against Pydantic (best-effort)
                        rows.append(Agent(**data).model_dump())
                        scraped_this_zip += 1
                        rand_sleep()
                    except Exception as e:
                        print(f"  profile failed: {prof} -> {e}")
                rand_sleep(1.5, 3.0)
            rand_sleep(2.5, 5.0)

        browser.close()

    write_rows_csv(rows, out_csv)
    print(f"Done. Wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    # Minimal dry-run caps: remove or raise for full crawl
    run(
        zip_csv="data/US zip-state-city.csv",
        zip_col="ZIP Code",
        pages_per_zip=2,
        limit_profiles_per_zip=10,
        out_csv="data/agents.csv"
    )

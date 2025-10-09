# simple_state_scrape.py
import os, json, time, requests
from urllib.parse import quote
from dotenv import load_dotenv

# --- load env ---
load_dotenv()  # loads variables from .env into process env

# --- config ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY_NEW")
if not RAPIDAPI_KEY:
    raise SystemExit("RAPIDAPI_KEY_NEW is not set. Put it in .env or the environment.")

HOST = "realtor-search.p.rapidapi.com"
BASE = f"https://{HOST}"
OUT_FILE = "agents_by_state.json"

# pace + retry
BASE_SLEEP = 1.0      # gentle base delay
MAX_RETRIES = 4
BACKOFF = 1.6

STATES = [
    'Alaska','Alabama','Arkansas','Arizona','California','Colorado','Connecticut',
    'District of Columbia','Delaware','Florida','Georgia','Hawaii','Iowa','Idaho',
    'Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland',
    'Maine','Michigan','Minnesota','Missouri','Mississippi','Montana','North Carolina',
    'North Dakota','Nebraska','New Hampshire','New Jersey','New Mexico','Nevada',
    'New York','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
    'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin',
    'West Virginia','Wyoming'
]

USPS = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
    "Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
    "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

session = requests.Session()
session.headers.update({
    "x-rapidapi-host": HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
})

def fetch_state_payload(usps_code: str):
    target = f"https://www.realtor.com/realestateagents/{usps_code}"
    url = f"{BASE}/agents/detail-url"
    params = {"url": quote(target, safe=":/")}

    delay = BASE_SLEEP
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                # too many requests or transient error - backoff and retry
                time.sleep(delay)
                delay *= BACKOFF
                continue
            # other errors: return body so you can inspect
            return {"error": f"HTTP {r.status_code}", "text": r.text}
        except requests.RequestException as e:
            # network hiccup; backoff and retry
            time.sleep(delay)
            delay *= BACKOFF

    return {"error": "max_retries_exceeded"}

def main():
    out = []
    for st in STATES:
        code = USPS.get(st)
        if not code:
            out.append({"state": st, "note": "no USPS code found"})
            continue

        payload = fetch_state_payload(code)
        out.append({
            "state": st,
            "usps": code,
            "source_url": f"https://www.realtor.com/realestateagents/{code}",
            "payload": payload
        })

        # small base delay between states to avoid 429
        time.sleep(BASE_SLEEP)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out)} state payloads to {OUT_FILE}")

if __name__ == "__main__":
    main()

from realtor_scraper import RealtorScraper, UserPreferences

scraper = RealtorScraper()

preferences = UserPreferences(
    city="Miami",
    state="FL",
    min_price=150000,
    max_price=250000,
    min_beds=2,
    max_beds=4,
    min_baths=2,
    property_type="single-family-home",
    sort_by="price-low-to-high"
)

properties = scraper.scrape_properties(preferences, max_pages=3)

# Save results
scraper.save_to_csv(properties, "austin_properties.csv")

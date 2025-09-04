-- Insert sample properties
INSERT INTO public.properties (title, description, price, address, city, state, zip_code, property_type, bedrooms, bathrooms, square_feet, lot_size, year_built) VALUES
('Modern Downtown Condo', 'Luxury condo in the heart of downtown with city views', 450000.00, '123 Main St Unit 15A', 'Austin', 'TX', '78701', 'condo', 2, 2.0, 1200, NULL, 2018),
('Suburban Family Home', 'Spacious 4-bedroom home in quiet neighborhood', 325000.00, '456 Oak Avenue', 'Austin', 'TX', '78745', 'house', 4, 3.0, 2400, 0.25, 2010),
('Investment Duplex', 'Great rental property with two units', 280000.00, '789 Pine Street', 'Austin', 'TX', '78702', 'house', 3, 2.0, 1800, 0.15, 1995),
('Luxury Townhouse', 'High-end townhouse with premium finishes', 520000.00, '321 Cedar Lane', 'Austin', 'TX', '78703', 'townhouse', 3, 2.5, 2000, 0.05, 2020),
('Commercial Office Space', 'Prime office location for business', 750000.00, '654 Business Blvd', 'Austin', 'TX', '78704', 'commercial', NULL, 2.0, 3500, 0.10, 2015);

-- Insert sample market analytics
INSERT INTO public.market_analytics (city, state, avg_price, median_price, price_per_sqft, market_trend, inventory_level, days_on_market, month_year) VALUES
('Austin', 'TX', 425000.00, 395000.00, 285.50, 'rising', 1250, 28, '2024-01-01'),
('Austin', 'TX', 435000.00, 405000.00, 292.75, 'rising', 1180, 25, '2024-02-01'),
('Austin', 'TX', 445000.00, 415000.00, 298.25, 'rising', 1100, 22, '2024-03-01'),
('Dallas', 'TX', 385000.00, 365000.00, 245.80, 'stable', 2100, 35, '2024-01-01'),
('Dallas', 'TX', 390000.00, 370000.00, 248.50, 'stable', 2050, 33, '2024-02-01'),
('Houston', 'TX', 295000.00, 275000.00, 185.25, 'declining', 3200, 45, '2024-01-01'),
('Houston', 'TX', 290000.00, 270000.00, 182.75, 'declining', 3350, 48, '2024-02-01');

# scorer.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class AgentScorer:
    def __init__(self):
        self.component_weights = {
            "performance_track_record": 0.35,
            "market_expertise":        0.25,
            "client_satisfaction":     0.20,
            "professional_standing":   0.12,
            "availability_resp":       0.08,
        }
        self.performance_sub = {"recent_deals":0.40,"total_exp":0.25,"tx_value":0.20,"consistency":0.15}
        self.expertise_sub   = {"geo":0.35,"ptype":0.30,"price":0.35}
        self.satisfaction_sub= {"rating":0.60,"reviews":0.40}

    # ---------- helpers ----------
    @staticmethod
    def _overlap_score(agent_vals: List[str], target_vals: List[str]) -> float:
        """Calculate overlap score between agent service areas and target areas with City, State support."""
        if not agent_vals or not target_vals: 
            return 0.0
        
        a = [s.lower().strip() for s in agent_vals if s]
        t = [s.lower().strip() for s in target_vals if s]
        
        if not a or not t:
            return 0.0
        
        hits = 0.0
        for target in t:
            best_match_score = 0.0
            
            # Extract target components
            target_city, target_state = AgentScorer._extract_location_components(target)
            
            for agent_area in a:
                agent_city, agent_state = AgentScorer._extract_location_components(agent_area)
                
                # Exact full match "City, State"
                if target == agent_area:
                    best_match_score = max(best_match_score, 1.0)
                    continue
                
                # City and State both match
                if (target_city and agent_city and target_city == agent_city and
                    target_state and agent_state and target_state == agent_state):
                    best_match_score = max(best_match_score, 1.0)
                    continue
                
                # City matches exactly
                if target_city and agent_city and target_city == agent_city:
                    best_match_score = max(best_match_score, 0.9)
                    continue
                
                # State matches exactly
                if target_state and agent_state and target_state == agent_state:
                    best_match_score = max(best_match_score, 0.7)
                    continue
                
                # Partial city match
                if (target_city and agent_city and 
                    (target_city in agent_city or agent_city in target_city)):
                    best_match_score = max(best_match_score, 0.6)
                    continue
                
                # State abbreviation matching
                if target_state and agent_area:
                    state_abbrevs = AgentScorer._get_state_abbreviations()
                    target_abbrev = AgentScorer._get_state_abbreviation(target_state, state_abbrevs)
                    if target_abbrev and target_abbrev in agent_area:
                        best_match_score = max(best_match_score, 0.5)
                        continue
                
                # Fallback partial match
                if target in agent_area or agent_area in target:
                    best_match_score = max(best_match_score, 0.4)
            
            hits += best_match_score
        
        return min(1.0, hits / len(t))

    @staticmethod
    def _extract_location_components(location: str) -> tuple[str, str]:
        """Extract city and state from 'City, State' format."""
        if ", " in location:
            parts = location.split(", ", 1)
            return parts[0].strip().lower(), parts[1].strip().lower()
        else:
            return location.strip().lower(), ""
    
    @staticmethod
    def _get_state_abbreviations() -> dict[str, str]:
        """Get mapping of full state names to abbreviations."""
        return {
            "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
            "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
            "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
            "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
            "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
            "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
            "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
            "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
            "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
            "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
            "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
            "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
            "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"
        }
    
    @staticmethod
    def _get_state_abbreviation(state_name: str, abbrev_map: dict[str, str]) -> str:
        """Get state abbreviation from full name."""
        return abbrev_map.get(state_name.lower(), "")

    @staticmethod
    def _enhanced_geographical_match(agent_data: pd.Series, target_locations: List[str]) -> float:
        """
        Enhanced geographical matching using multiple location fields.
        Prioritizes comprehensive_service_areas, then falls back to other fields.
        """
        if not target_locations:
            return 0.8  # Neutral score when no specific location requested
        
        # Get agent's geographical data
        comprehensive_areas = agent_data.get("comprehensive_service_areas", [])
        primary_regions = agent_data.get("primary_service_regions", [])
        business_market = agent_data.get("business_market_normalized", "")
        office_state = agent_data.get("office_state", "")
        
        # Convert to lists if they're strings (from database)
        if isinstance(comprehensive_areas, str):
            comprehensive_areas = [comprehensive_areas] if comprehensive_areas else []
        if isinstance(primary_regions, str):
            primary_regions = [primary_regions] if primary_regions else []
        
        scores = []
        
        # 1. Check comprehensive service areas (highest priority)
        if comprehensive_areas:
            score = AgentScorer._overlap_score(comprehensive_areas, target_locations)
            scores.append(("comprehensive", score, 1.0))
        
        # 2. Check primary service regions
        if primary_regions:
            score = AgentScorer._overlap_score(primary_regions, target_locations)
            scores.append(("primary", score, 0.9))
        
        # 3. Check business market
        if business_market and business_market != "Unknown":
            score = AgentScorer._overlap_score([business_market], target_locations)
            scores.append(("business_market", score, 0.8))
        
        # 4. Check office state
        if office_state and office_state != "Unknown":
            score = AgentScorer._overlap_score([office_state], target_locations)
            scores.append(("office_state", score, 0.7))
        
        if not scores:
            return 0.2  # Low score if no geographical data available
        
        # Return the highest weighted score
        return max(score * weight for _, score, weight in scores)

    def _price_match_series(self, df: pd.DataFrame, min_price: float, max_price: float) -> pd.Series:
        center = (min_price + max_price)/2
        span   = max(1.0, max_price - min_price)
        def f(row):
            med, mn, mx = row["deal_prices_median"], row["deal_prices_min"], row["deal_prices_max"]
            if med == 0: return 0.2
            if mn <= min_price and mx >= max_price: return 1.0
            if min_price <= med <= max_price: return 0.9
            dist = abs(med - center)/(2*span)
            return max(0.1, 1 - dist)
        return df.apply(f, axis=1)

    # ---------- component scores ----------
    def performance(self, df: pd.DataFrame) -> pd.Series:
        max_recent = max(df["past_year_deals"].max(), 1)
        recent = df["past_year_deals"]/max_recent
        exp    = (df["experience_score"]/max(df["experience_score"].max(),1e-9)).fillna(0)
        max_vol= max(df["transaction_volume_lifetime"].max(), 1)
        vol    = df["transaction_volume_lifetime"]/max_vol
        cons   = df["recent_activity_ratio"].clip(0,1)
        return (self.performance_sub["recent_deals"]*recent +
                self.performance_sub["total_exp"]  *exp   +
                self.performance_sub["tx_value"]   *vol   +
                self.performance_sub["consistency"]*cons)

    def expertise(self, df: pd.DataFrame,
                  regions: Optional[List[str]], ptypes: Optional[List[str]],
                  price_range: Optional[Tuple[float,float]]) -> pd.Series:
        
        # Enhanced geographical scoring
        if regions:
            geo = df.apply(lambda row: self._enhanced_geographical_match(row, regions), axis=1)
        else:
            # If no regions specified, favor agents with broader coverage
            geo = (df["num_comprehensive_areas"]/max(df["num_comprehensive_areas"].max(),1)).clip(0,1)
        
        # Property type matching (unchanged)
        if ptypes:
            prop = df["property_types"].apply(lambda p: self._overlap_score(p, ptypes))
        else:
            prop = 1 - (df["specialization_index"] - 0.5).abs()
        
        # Price range matching (unchanged)
        if price_range:
            price = self._price_match_series(df, price_range[0], price_range[1])
        else:
            price = (1 - df["price_coefficient_variation"].clip(0,1))
        
        return (self.expertise_sub["geo"]*geo +
                self.expertise_sub["ptype"]*prop +
                self.expertise_sub["price"]*price)

    def satisfaction(self, df: pd.DataFrame) -> pd.Series:
        rating = (df["star_rating"]/5.0).clip(0,1)
        max_rev= max(df["num_reviews"].max(),1)
        rev    = np.sqrt(df["num_reviews"]/max_rev)
        return self.satisfaction_sub["rating"]*rating + self.satisfaction_sub["reviews"]*rev

    def professional(self, df: pd.DataFrame) -> pd.Series:
        prem = df["is_premier"].astype(float)
        team = df["partner"].astype(float)*0.5
        active = df["is_active"].astype(float)
        brokerage = (df["brokerage_name"].fillna("").ne("Independent")).astype(float)*0.3
        return np.minimum(1.0, 0.4*prem + 0.3*active + 0.2*team + 0.1*brokerage)

    def availability(self, df: pd.DataFrame) -> pd.Series:
        phone = (df["phone_number"]!="Not Available").astype(float)
        contact = df["profile_contact_enabled"].astype(float)
        recent  = (df["past_year_deals"]>0).astype(float)
        active  = df["is_active"].astype(float)
        return 0.3*phone + 0.2*contact + 0.3*recent + 0.2*active

    # ---------- total ----------
    def total(self, df: pd.DataFrame,
              regions: Optional[List[str]],
              ptypes:  Optional[List[str]],
              price_range: Optional[Tuple[float,float]]) -> tuple[pd.Series, pd.DataFrame]:
        perf = self.performance(df)
        exp  = self.expertise(df, regions, ptypes, price_range)
        sat  = self.satisfaction(df)
        prof = self.professional(df)
        avail= self.availability(df)
        total = (self.component_weights["performance_track_record"]*perf +
                 self.component_weights["market_expertise"]       *exp  +
                 self.component_weights["client_satisfaction"]    *sat  +
                 self.component_weights["professional_standing"]  *prof +
                 self.component_weights["availability_resp"]      *avail)
        breakdown = pd.DataFrame({
            "performance_score": perf, "expertise_score": exp,
            "satisfaction_score": sat, "professional_score": prof,
            "availability_score": avail, "total_score": total
        })
        return total, breakdown
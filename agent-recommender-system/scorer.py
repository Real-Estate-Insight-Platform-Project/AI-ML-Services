# scorer.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

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
        if not agent_vals or not target_vals: return 0.0
        a = [s.lower().strip() for s in agent_vals]
        t = [s.lower().strip() for s in target_vals]
        hits = sum(1 for x in t if any(x in y or y in x for y in a))
        return min(1.0, hits / len(t))

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
                  regions: List[str]|None, ptypes: List[str]|None,
                  price_range: Tuple[float,float]|None) -> pd.Series:
        if regions:
            geo = df["primary_service_regions"].apply(lambda r: self._overlap_score(r, regions))
        else:
            geo = (df["num_service_regions"]/max(df["num_service_regions"].max(),1)).clip(0,1)
        if ptypes:
            prop = df["property_types"].apply(lambda p: self._overlap_score(p, ptypes))
        else:
            prop = 1 - (df["specialization_index"] - 0.5).abs()
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
              regions: List[str]|None,
              ptypes:  List[str]|None,
              price_range: Tuple[float,float]|None) -> tuple[pd.Series, pd.DataFrame]:
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

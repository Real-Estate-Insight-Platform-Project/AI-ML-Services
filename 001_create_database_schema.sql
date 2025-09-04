-- Create user profiles table
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  full_name TEXT,
  user_role TEXT NOT NULL CHECK (user_role IN ('buyer', 'investor')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for profiles
CREATE POLICY "profiles_select_own" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "profiles_insert_own" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "profiles_update_own" ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "profiles_delete_own" ON public.profiles FOR DELETE USING (auth.uid() = id);

-- Create properties table
CREATE TABLE IF NOT EXISTS public.properties (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  description TEXT,
  price DECIMAL(12,2) NOT NULL,
  address TEXT NOT NULL,
  city TEXT NOT NULL,
  state TEXT NOT NULL,
  zip_code TEXT NOT NULL,
  property_type TEXT NOT NULL CHECK (property_type IN ('house', 'apartment', 'condo', 'townhouse', 'commercial')),
  bedrooms INTEGER,
  bathrooms DECIMAL(3,1),
  square_feet INTEGER,
  lot_size DECIMAL(10,2),
  year_built INTEGER,
  listing_status TEXT NOT NULL DEFAULT 'active' CHECK (listing_status IN ('active', 'pending', 'sold', 'off_market')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on properties (public read access for now)
ALTER TABLE public.properties ENABLE ROW LEVEL SECURITY;
CREATE POLICY "properties_select_all" ON public.properties FOR SELECT TO authenticated USING (true);

-- Create market analytics table
CREATE TABLE IF NOT EXISTS public.market_analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  city TEXT NOT NULL,
  state TEXT NOT NULL,
  avg_price DECIMAL(12,2),
  median_price DECIMAL(12,2),
  price_per_sqft DECIMAL(8,2),
  market_trend TEXT CHECK (market_trend IN ('rising', 'stable', 'declining')),
  inventory_level INTEGER,
  days_on_market INTEGER,
  month_year DATE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on market analytics
ALTER TABLE public.market_analytics ENABLE ROW LEVEL SECURITY;
CREATE POLICY "market_analytics_select_all" ON public.market_analytics FOR SELECT TO authenticated USING (true);

-- Create user favorites table
CREATE TABLE IF NOT EXISTS public.user_favorites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  property_id UUID NOT NULL REFERENCES public.properties(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, property_id)
);

-- Enable RLS on user favorites
ALTER TABLE public.user_favorites ENABLE ROW LEVEL SECURITY;
CREATE POLICY "favorites_select_own" ON public.user_favorites FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "favorites_insert_own" ON public.user_favorites FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "favorites_delete_own" ON public.user_favorites FOR DELETE USING (auth.uid() = user_id);

-- Create investment analysis table
CREATE TABLE IF NOT EXISTS public.investment_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  property_id UUID NOT NULL REFERENCES public.properties(id) ON DELETE CASCADE,
  purchase_price DECIMAL(12,2) NOT NULL,
  down_payment DECIMAL(12,2) NOT NULL,
  loan_amount DECIMAL(12,2) NOT NULL,
  interest_rate DECIMAL(5,3) NOT NULL,
  loan_term INTEGER NOT NULL,
  monthly_rent DECIMAL(8,2),
  monthly_expenses DECIMAL(8,2),
  cash_flow DECIMAL(8,2),
  cap_rate DECIMAL(5,3),
  roi DECIMAL(5,3),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on investment analysis
ALTER TABLE public.investment_analysis ENABLE ROW LEVEL SECURITY;
CREATE POLICY "investment_analysis_select_own" ON public.investment_analysis FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "investment_analysis_insert_own" ON public.investment_analysis FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "investment_analysis_update_own" ON public.investment_analysis FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "investment_analysis_delete_own" ON public.investment_analysis FOR DELETE USING (auth.uid() = user_id);

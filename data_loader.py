import pandas as pd
import numpy as np

class MarketDataLoader:
    def __init__(self, base_dir):
        """Loads the parquets into RAM once upon initialization."""
        print("Loading Options, Spot, Yield, and Dividend Data into memory...")
        
        self.df_options = pd.read_parquet(f"{base_dir}\\SPX_OptionPrices_Cleaned.parquet")
        self.df_spot    = pd.read_parquet(f"{base_dir}\\SPX_IndexPrices.parquet")
        self.df_yield   = pd.read_parquet(f"{base_dir}\\ZeroCouponYieldCurve.parquet")
        self.df_div     = pd.read_parquet(f"{base_dir}\\SPX_IndexDividendYields.parquet")

        self.df_spot['date_str']  = self.df_spot['date'].dt.strftime('%Y-%m-%d')
        self.df_yield['date_str'] = self.df_yield['date'].dt.strftime('%Y-%m-%d')
        self.df_div['date_str']   = self.df_div['date'].dt.strftime('%Y-%m-%d')
        self.df_options['date_str'] = self.df_options['date'].dt.strftime('%Y-%m-%d')
        self.df_options['exdate_str'] = self.df_options['exdate'].dt.strftime('%Y-%m-%d')

        self.spot_dict = dict(zip(self.df_spot['date_str'], self.df_spot['close'])) 
        print("✅ Data Loaded Successfully.")

    def get_market_state(self, target_date, target_exdate, strike_bound_pct=0.15):
        """Extracts S0, r, q, T and a trimmed option chain for specific dates."""
        
        days_to_maturity = (pd.to_datetime(target_exdate) - pd.to_datetime(target_date)).days
        T = days_to_maturity / 365.0

        # 1. Spot Price
        if target_date not in self.spot_dict:
            raise ValueError(f"❌ Could not find SPX Spot Price for {target_date}.")
        S0 = self.spot_dict[target_date]

        # 2. Risk-Free Rate
        daily_yield_curve = self.df_yield[self.df_yield['date_str'] == target_date].sort_values('days')
        if not daily_yield_curve.empty:
            r = np.interp(days_to_maturity, daily_yield_curve['days'], daily_yield_curve['rate']) / 100.0 
        else:
            raise ValueError(f"❌ Missing Yield Curve data for {target_date}")

        # 3. Dividend Yield
        daily_div = self.df_div[self.df_div['date_str'] == target_date]
        if not daily_div.empty:
            if 'days' in daily_div.columns:
                daily_div_curve = daily_div.sort_values('days')
                q = np.interp(days_to_maturity, daily_div_curve['days'], daily_div_curve['rate']) / 100.0
            elif 'rate' in daily_div.columns:
                q = daily_div['rate'].iloc[0] / 100.0
            else:
                q = 0.0
        else:
            q = 0.0

        # 4. Extract Option Chain
        df_slice = self.df_options[(self.df_options['date_str'] == target_date) & 
                                   (self.df_options['exdate_str'] == target_exdate)].copy()

        calls = df_slice[(df_slice['cp_flag'] == 'C') & (df_slice['strike_price'] >= S0)].copy()
        chain = calls.sort_values('strike_price')
        chain['mid_price'] = (chain['best_bid'] + chain['best_offer']) / 2.0
        chain = chain[chain['best_bid'] > 0] 

        # 5. Trim Bounds (WRDS Format Fix Included)
        lower_bound = (S0 * (1 - strike_bound_pct)) * 1000 
        upper_bound = (S0 * (1 + strike_bound_pct)) * 1000
        chain = chain[(chain['strike_price'] >= lower_bound) & (chain['strike_price'] <= upper_bound)]

        market_strikes = chain['strike_price'].values / 1000.0 
        market_prices = chain['mid_price'].values

        return {
            'S0': S0, 'T': T, 'r': r, 'q': q, 
            'strikes': market_strikes, 'prices': market_prices
        }
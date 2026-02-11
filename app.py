import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as si
import plotly.graph_objects as go
from abc import ABC, abstractmethod

# ==========================================
# 1. CORE ANALYTICS ( The Math Engine )
# ==========================================

class Instrument(ABC):
    """
    Abstract Base Class for all financial instruments.
    """
    def __init__(self, position=1.0):
        self.position = position  # 1.0 for Long, -1.0 for Short

    @abstractmethod
    def price(self, S, T, r, sigma):
        """Calculate theoretical price today (t=0)"""
        pass

    @abstractmethod
    def payoff(self, S):
        """Calculate payoff at maturity (t=T)"""
        pass
    
    @abstractmethod
    def name(self):
        pass

class ZeroCouponBond(Instrument):
    def __init__(self, face_value, position=1.0):
        super().__init__(position)
        self.face_value = face_value

    def price(self, S, T, r, sigma):
        # Calculate the scalar bond price
        P = self.position * self.face_value * np.exp(-r * T)
        # Return array if S is array
        if isinstance(S, np.ndarray):
            return np.full_like(S, P, dtype=float)
        else:
            return P

    def payoff(self, S):
        return np.full_like(S, self.position * self.face_value)

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} Zero Bond (Face: {self.face_value})"

class VanillaOption(Instrument):
    def __init__(self, K, option_type='call', position=1.0):
        super().__init__(position)
        self.K = K
        self.option_type = option_type.lower()

    def price(self, S, T, r, sigma):
        if T <= 1e-6:
            return self.payoff(S)

        # Safety for log(0)
        S_safe = np.maximum(S, 1e-9) if isinstance(S, np.ndarray) else max(S, 1e-9)

        d1 = (np.log(S_safe / self.K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            val = (S_safe * si.norm.cdf(d1) - self.K * np.exp(-r * T) * si.norm.cdf(d2))
        else:
            val = (self.K * np.exp(-r * T) * si.norm.cdf(-d2) - S_safe * si.norm.cdf(-d1))
        
        return self.position * val

    def payoff(self, S):
        if self.option_type == 'call':
            return self.position * np.maximum(S - self.K, 0)
        else:
            return self.position * np.maximum(self.K - S, 0)

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} {self.option_type.capitalize()} (K={self.K})"

class DigitalOption(Instrument):
    def __init__(self, K, payout=1.0, option_type='call', position=1.0):
        super().__init__(position)
        self.K = K
        self.payout = payout
        self.option_type = option_type.lower()

    def price(self, S, T, r, sigma):
        if T <= 1e-6:
            return self.payoff(S)

        S_safe = np.maximum(S, 1e-9) if isinstance(S, np.ndarray) else max(S, 1e-9)
        d1 = (np.log(S_safe / self.K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            val = np.exp(-r * T) * si.norm.cdf(d2) * self.payout
        else:
            val = np.exp(-r * T) * si.norm.cdf(-d2) * self.payout
        
        return self.position * val

    def payoff(self, S):
        if self.option_type == 'call':
            return self.position * np.where(S > self.K, self.payout, 0.0)
        else:
            return self.position * np.where(S < self.K, self.payout, 0.0)

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} Digital {self.option_type.capitalize()} (K={self.K}, Pay={self.payout})"

class BarrierOption(Instrument):
    def __init__(self, K, H, option_type='call', barrier_type='up-out', position=1.0):
        super().__init__(position)
        self.K = K
        self.H = H
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()

    def price(self, S, T, r, sigma):
        if T <= 1e-6: return self.payoff(S)
        
        # --- Helper for standard BS Call/Put ---
        def bs_price(s_val, k_val, t_val, r_val, v_val, opt_type):
            s_safe = np.maximum(s_val, 1e-9) 
            d1 = (np.log(s_safe / k_val) + (r_val + 0.5 * v_val**2) * t_val) / (v_val * np.sqrt(t_val))
            d2 = d1 - v_val * np.sqrt(t_val)
            if opt_type == 'call':
                return s_safe * si.norm.cdf(d1) - k_val * np.exp(-r_val * t_val) * si.norm.cdf(d2)
            else:
                return k_val * np.exp(-r_val * t_val) * si.norm.cdf(-d2) - s_safe * si.norm.cdf(-d1)

        # Reflection Principle logic
        lam = (r + 0.5 * sigma**2) / sigma**2
        vanilla = bs_price(S, self.K, T, r, sigma, self.option_type)
        
        S_safe = np.maximum(S, 1e-9)
        reflection_factor = (self.H / S_safe) ** (2 * lam - 2)
        S_reflected = (self.H ** 2) / S_safe
        vanilla_reflected = bs_price(S_reflected, self.K, T, r, sigma, self.option_type)
        knock_in_value = reflection_factor * vanilla_reflected
        
        price = np.zeros_like(S) if isinstance(S, np.ndarray) else 0.0

        if self.barrier_type == 'up-out':
            p = vanilla - knock_in_value
            if not isinstance(S, np.ndarray):
                if S >= self.H: return 0.0
                return self.position * p
            p[S >= self.H] = 0.0
            return self.position * p

        elif self.barrier_type == 'down-out':
            p = vanilla - knock_in_value
            if not isinstance(S, np.ndarray):
                if S <= self.H: return 0.0
                return self.position * p
            p[S <= self.H] = 0.0
            return self.position * p
            
        return 0.0

    def payoff(self, S):
        if self.option_type == 'call':
            p = np.maximum(S - self.K, 0)
        else:
            p = np.maximum(self.K - S, 0)
        
        if self.barrier_type == 'up-out':
            p[S >= self.H] = 0
        elif self.barrier_type == 'down-out':
            p[S <= self.H] = 0
            
        return self.position * p

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} {self.barrier_type} {self.option_type} (K={self.K}, H={self.H})"


# ==========================================
# 2. RISK ENGINE (GREEKS)
# ==========================================

def calculate_portfolio_greeks(portfolio, S, T, r, sigma, skew_active=False, skew_slope=0.0, spot_ref=100.0):
    """
    Calculates Greeks using Bumping.
    Updated to handle Skew logic inside the bumping.
    """
    
    # Helper to calculate Price WITH Skew logic
    def get_price_skewed(p_list, s_val, t_val, r_val, v_base):
        total = 0.0
        for leg in p_list:
            leg_sigma = v_base
            if skew_active and hasattr(leg, 'K'):
                # Apply skew based on Moneyness (using Reference Spot for consistency)
                moneyness = (spot_ref - leg.K) / spot_ref
                leg_sigma = v_base + (skew_slope * moneyness)
                leg_sigma = max(0.01, leg_sigma)
            
            total += leg.price(s_val, t_val, r_val, leg_sigma)
        return total

    # 1. Base Price
    base_price = get_price_skewed(portfolio, S, T, r, sigma)

    # 2. Delta & Gamma (Bump Spot 1%)
    dS = S * 0.01 
    p_up = get_price_skewed(portfolio, S + dS, T, r, sigma)
    p_down = get_price_skewed(portfolio, S - dS, T, r, sigma)
    
    delta = (p_up - p_down) / (2 * dS)
    gamma = (p_up - 2 * base_price + p_down) / (dS ** 2)

    # 3. Vega (Bump Vol 1%)
    dVol = 0.01
    v_up = get_price_skewed(portfolio, S, T, r, sigma + dVol)
    v_down = get_price_skewed(portfolio, S, T, r, sigma - dVol)
    vega = (v_up - v_down) / (2 * dVol) 

    # 4. Theta (Bump Time 1 Day)
    dt = 1 / 365.0
    if T > dt:
        p_tomorrow = get_price_skewed(portfolio, S, T - dt, r, sigma)
        theta = (p_tomorrow - base_price)
    else:
        theta = 0.0

    # 5. Rho (Bump Rate 1%)
    dr = 0.01
    r_up = get_price_skewed(portfolio, S, T, r + dr, sigma)
    r_down = get_price_skewed(portfolio, S, T, r - dr, sigma)
    rho = (r_up - r_down) / (2 * dr)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega / 100,
        "Theta": theta,
        "Rho": rho / 100
    }


# ==========================================
# 3. STREAMLIT FRONTEND
# ==========================================

st.set_page_config(page_title="Structurer's Workbench", layout="wide")
st.title("🛠️ Derivatives Structuring Workbench")

# --- Initialize Session State for TWO Portfolios ---
if 'portfolio_a' not in st.session_state:
    st.session_state.portfolio_a = []
if 'portfolio_b' not in st.session_state:
    st.session_state.portfolio_b = []

# --- Sidebar: Market Conditions ---
st.sidebar.header("1. Market Conditions")
spot_ref = st.sidebar.number_input("Reference Spot Price", value=100.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, format="%.2f")
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.20)
T = st.sidebar.slider("Time to Maturity (Years)", 0.01, 2.0, 1.0)

# --- Sidebar: Advanced Settings ---
st.sidebar.markdown("---")
st.sidebar.header("2. Advanced Models")
use_skew = st.sidebar.checkbox("Enable Volatility Skew")
skew_slope = 0.0
if use_skew:
    skew_slope = st.sidebar.slider("Skew Slope", 0.0, 0.5, 0.2, help="OTM Puts get more expensive.")

# --- Sidebar: Portfolio Management ---
st.sidebar.markdown("---")
st.sidebar.header("3. Manage Portfolios")
compare_mode = st.sidebar.checkbox("Compare Mode (A vs B)")

if compare_mode:
    active_port_name = st.sidebar.radio("Select Portfolio to Edit:", ["Portfolio A", "Portfolio B"])
    # Point active_portfolio to the list in session state
    if active_port_name == "Portfolio A":
        active_portfolio = st.session_state.portfolio_a
    else:
        active_portfolio = st.session_state.portfolio_b
else:
    active_portfolio = st.session_state.portfolio_a
    active_port_name = "Portfolio A"
    st.sidebar.caption("Currently editing Portfolio A")

# --- Sidebar: Add Instruments ---
st.sidebar.markdown("---")
st.sidebar.header("4. Add Legs")

inst_type = st.sidebar.selectbox("Instrument Type", ["Vanilla Option", "Digital Option", "Barrier Option", "Zero Coupon Bond"])
position_side = st.sidebar.selectbox("Position", ["Long (+1)", "Short (-1)"])
pos_val = 1.0 if position_side == "Long (+1)" else -1.0

# Add Leg Logic
if inst_type == "Vanilla Option":
    opt_type = st.sidebar.radio("Option Type", ["Call", "Put"])
    strike = st.sidebar.number_input("Strike Price (K)", value=100.0)
    if st.sidebar.button(f"Add to {active_port_name}"):
        active_portfolio.append(VanillaOption(strike, opt_type, pos_val))

elif inst_type == "Digital Option":
    opt_type = st.sidebar.radio("Type", ["Call (Pay if S > K)", "Put (Pay if S < K)"])
    strike = st.sidebar.number_input("Strike Price (K)", value=100.0)
    payout = st.sidebar.number_input("Payout ($)", value=1.0)
    if st.sidebar.button(f"Add to {active_port_name}"):
        clean_type = "call" if "Call" in opt_type else "put"
        active_portfolio.append(DigitalOption(strike, payout, clean_type, pos_val))

elif inst_type == "Barrier Option":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        opt_type = st.radio("Option Type", ["Call", "Put"])
    with col2:
        bar_type = st.radio("Barrier Type", ["Up-Out", "Down-Out"])
    strike = st.sidebar.number_input("Strike Price (K)", value=100.0)
    barrier = st.sidebar.number_input("Barrier Level (H)", value=120.0)
    if st.sidebar.button(f"Add to {active_port_name}"):
        o_type = "call" if opt_type == "Call" else "put"
        b_type = "up-out" if bar_type == "Up-Out" else "down-out"
        active_portfolio.append(BarrierOption(strike, barrier, o_type, b_type, pos_val))

elif inst_type == "Zero Coupon Bond":
    face = st.sidebar.number_input("Face Value", value=100.0)
    if st.sidebar.button(f"Add to {active_port_name}"):
        active_portfolio.append(ZeroCouponBond(face, pos_val))

# --- Main Area: Display Active Portfolio Legs ---
st.subheader(f"Current Legs: {active_port_name}")

if active_portfolio:
    for i, leg in enumerate(active_portfolio):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**Leg {i+1}:** {leg.name()}")
        with col2:
            if st.button("Remove", key=f"del_{active_port_name}_{i}"):
                active_portfolio.pop(i)
                st.rerun()
else:
    st.info("No legs in this portfolio.")

# ==========================================
# 4. VISUALIZATION ENGINE
# ==========================================

st.markdown("---")
st.header("Structure Analysis")

# Helper to calculate price arrays for plotting
def get_plot_values(portfolio, S_arr):
    price_arr = np.zeros_like(S_arr)
    payoff_arr = np.zeros_like(S_arr)
    
    for leg in portfolio:
        # Skew Logic
        leg_sigma = sigma
        if use_skew and hasattr(leg, 'K'):
            moneyness = (spot_ref - leg.K) / spot_ref
            leg_sigma = sigma + (skew_slope * moneyness)
            leg_sigma = max(0.01, leg_sigma)
            
        price_arr += leg.price(S_arr, T, r, leg_sigma)
        payoff_arr += leg.payoff(S_arr)
    return price_arr, payoff_arr

spot_range = np.linspace(spot_ref * 0.5, spot_ref * 1.5, 200)

fig = go.Figure()

# 1. Plot Portfolio A
price_a, payoff_a = get_plot_values(st.session_state.portfolio_a, spot_range)
fig.add_trace(go.Scatter(x=spot_range, y=price_a, mode='lines', name='Price A (Today)', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=spot_range, y=payoff_a, mode='lines', name='Payoff A (Expiry)', line=dict(color='lightblue', dash='dash')))

# 2. Plot Portfolio B (If Compare Mode)
if compare_mode:
    price_b, payoff_b = get_plot_values(st.session_state.portfolio_b, spot_range)
    fig.add_trace(go.Scatter(x=spot_range, y=price_b, mode='lines', name='Price B (Today)', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=payoff_b, mode='lines', name='Payoff B (Expiry)', line=dict(color='pink', dash='dash')))

fig.add_vline(x=spot_ref, line_dash="dot", line_color="gray", annotation_text="Spot")
fig.update_layout(title="Payoff & Price Analysis", xaxis_title="Spot Price", yaxis_title="Value ($)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- Risk Metrics (Greeks) ---
st.subheader("Risk Metrics (Greeks)")

if compare_mode:
    # Compare Mode: Side-by-Side Table
    g_a = calculate_portfolio_greeks(st.session_state.portfolio_a, spot_ref, T, r, sigma, use_skew, skew_slope, spot_ref)
    g_b = calculate_portfolio_greeks(st.session_state.portfolio_b, spot_ref, T, r, sigma, use_skew, skew_slope, spot_ref)
    
    df_greeks = pd.DataFrame([g_a, g_b], index=["Portfolio A", "Portfolio B"])
    st.table(df_greeks.style.format("{:.4f}"))

else:
    # Single Mode: Beautiful Cards
    g = calculate_portfolio_greeks(st.session_state.portfolio_a, spot_ref, T, r, sigma, use_skew, skew_slope, spot_ref)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Delta", f"{g['Delta']:.3f}", help="Shares to hedge")
    c2.metric("Gamma", f"{g['Gamma']:.4f}", help="Convexity")
    c3.metric("Vega", f"{g['Vega']:.3f}", help="Vol Sensitivity")
    c4.metric("Theta", f"{g['Theta']:.3f}", help="Daily Decay")
    c5.metric("Rho", f"{g['Rho']:.3f}", help="Rate Sensitivity")
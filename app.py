import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as si
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from math import factorial
import cmath
from scipy.integrate import quad

# ==========================================
# 0. CONFIG & SHARED STATE
# ==========================================
st.set_page_config(page_title="Quant Structuring Desk", layout="wide")
st.title("⚡ Quantitative Derivatives Workbench")

# Initialize Session State for Portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# ==========================================
# 1. CORE MATH CLASSES (YOUR ENGINE)
# ==========================================

class Instrument(ABC):
    def __init__(self, position=1.0, sigma_override=None):
        self.position = position 
        self.sigma_override = sigma_override # Local Volatility Override

    def get_sigma(self, global_sigma):
        return self.sigma_override if self.sigma_override is not None else global_sigma

    @abstractmethod
    def price(self, S, T, r, sigma):
        pass

    @abstractmethod
    def payoff(self, S):
        pass
    
    @abstractmethod
    def name(self):
        pass

    # --- FINITE DIFFERENCE GREEKS ---
    def delta(self, S, T, r, sigma):
        dS = S * 0.001
        up = self.price(S + dS, T, r, sigma)
        dn = self.price(S - dS, T, r, sigma)
        return (up - dn) / (2 * dS)

    def gamma(self, S, T, r, sigma):
        dS = S * 0.001
        up = self.price(S + dS, T, r, sigma)
        mid = self.price(S, T, r, sigma)
        dn = self.price(S - dS, T, r, sigma)
        return (up - 2 * mid + dn) / (dS ** 2)

    def vega(self, S, T, r, sigma):
        dSig = 0.001
        # Temporarily force the override to bump the active volatility
        original_override = self.sigma_override
        base_sig = self.get_sigma(sigma)
        
        self.sigma_override = base_sig + dSig
        up = self.price(S, T, r, sigma)
        self.sigma_override = base_sig - dSig
        dn = self.price(S, T, r, sigma)
        
        self.sigma_override = original_override # Restore state
        return (up - dn) / (2 * dSig) / 100 # Scaled to 1% vol change

    def theta(self, S, T, r, sigma):
        dT = 1 / 365.0 # 1 Day decay
        if T <= dT: return 0.0
        # Theta is time decay, so Price(T - 1 day) - Price(T)
        return (self.price(S, T - dT, r, sigma) - self.price(S, T, r, sigma))

class Stock(Instrument):
    def __init__(self, position=1.0, sigma_override=None):
        super().__init__(position, sigma_override)

    def price(self, S, T, r, sigma):
        return self.position * S

    def payoff(self, S):
        return self.position * S

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} Stock"

class ZeroCouponBond(Instrument):
    def __init__(self, face_value, position=1.0, sigma_override=None):
        super().__init__(position, sigma_override)
        self.face_value = face_value

    def price(self, S, T, r, sigma):
        P = self.position * self.face_value * np.exp(-r * T)
        if isinstance(S, np.ndarray):
            return np.full_like(S, P, dtype=float)
        return P

    def payoff(self, S):
        return np.full_like(S, self.position * self.face_value) if isinstance(S, np.ndarray) else self.position * self.face_value

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} Zero Bond (Face: {self.face_value})"

class VanillaOption(Instrument):
    def __init__(self, K, option_type='call', position=1.0, sigma_override=None):
        super().__init__(position, sigma_override)
        self.K = K
        self.option_type = option_type.lower()

    def price(self, S, T, r, sigma):
        active_sigma = self.get_sigma(sigma) # Inject Local Vol
        if T <= 1e-6: return self.payoff(S)
        S_safe = np.maximum(S, 1e-9) if isinstance(S, np.ndarray) else max(S, 1e-9)
        d1 = (np.log(S_safe / self.K) + (r + 0.5 * active_sigma ** 2) * T) / (active_sigma * np.sqrt(T))
        d2 = d1 - active_sigma * np.sqrt(T)
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
    def __init__(self, K, payout=1.0, option_type='call', position=1.0, sigma_override=None):
        super().__init__(position, sigma_override)
        self.K = K
        self.payout = payout
        self.option_type = option_type.lower()

    def price(self, S, T, r, sigma):
        active_sigma = self.get_sigma(sigma) # Inject Local Vol
        if T <= 1e-6: return self.payoff(S)
        S_safe = np.maximum(S, 1e-9) if isinstance(S, np.ndarray) else max(S, 1e-9)
        d1 = (np.log(S_safe / self.K) + (r + 0.5 * active_sigma ** 2) * T) / (active_sigma * np.sqrt(T))
        d2 = d1 - active_sigma * np.sqrt(T)
        if self.option_type == 'call':
            val = np.exp(-r * T) * si.norm.cdf(d2) * self.payout
        else:
            val = np.exp(-r * T) * si.norm.cdf(-d2) * self.payout
        return self.position * val

    def payoff(self, S):
        if isinstance(S, np.ndarray):
            if self.option_type == 'call':
                return self.position * np.where(S > self.K, self.payout, 0.0)
            else:
                return self.position * np.where(S < self.K, self.payout, 0.0)
        else:
            if self.option_type == 'call':
                return self.position * self.payout if S > self.K else 0.0
            return self.position * self.payout if S < self.K else 0.0

    def name(self):
        side = "Long" if self.position > 0 else "Short"
        return f"{side} Digital {self.option_type.capitalize()} (K={self.K})"

# ==========================================
# 2. PRICING MODELS (Merton Jump Diffusion)
# ==========================================

def merton_jump_diffusion_price(instrument, S, T, r, sigma, m_lam, m_gamma, m_delta):
    """
    Prices an instrument using Merton Jump Diffusion via infinite series approximation.
    """
    # FIX: Use string comparison to avoid Streamlit class-redefinition issues
    if instrument.__class__.__name__ != "VanillaOption":
        # Fallback for non-vanilla instruments (like Digital/Bond) to BSM
        return instrument.price(S, T, r, sigma)

    # --- Merton Logic Remains the Same ---
    k = np.exp(m_gamma + 0.5 * m_delta**2) - 1 
    lambda_prime = m_lam * (1 + k)
    
    price_merton = 0.0
    
    # Sum the first 15 terms
    for n in range(15):
        r_n = r - m_lam * k + (n * np.log(1 + k)) / T
        sigma_n = np.sqrt(sigma**2 + (n * m_delta**2) / T)
        
        prob_n_jumps = (np.exp(-lambda_prime * T) * (lambda_prime * T)**n) / factorial(n)
        
        # We call the standard BS price, but with ADJUSTED r_n and sigma_n
        bs_price = instrument.price(S, T, r_n, sigma_n)
        
        price_merton += prob_n_jumps * bs_price
        
    return price_merton

class HestonPricer:
    """
    Computes European Option Prices under the Heston Stochastic Volatility Model
    using the Gil-Pelaez Fourier inversion formula.
    """
    def __init__(self, S0, K, T, r, v0, kappa, theta, sigma, rho):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0       # Initial variance
        self.kappa = kappa # Mean reversion speed
        self.theta = theta # Long-run variance
        self.sigma = sigma # Vol of vol
        self.rho = rho     # Correlation (Stock vs Vol)

    def heston_char_func(self, phi):
        # Stable "Albrecher" formulation to avoid branch cuts
        # Constants
        prod = self.rho * self.sigma * 1j * phi
        
        # d calculation (Same as before)
        d_num = (prod - self.kappa)**2 + self.sigma**2 * (1j * phi + phi**2)
        d = cmath.sqrt(d_num)
        
        # g calculation (The key difference is here)
        # We use the auxiliary variable 'g' differently to avoid denominator -> 0
        g_num = self.kappa - prod - d
        g_den = self.kappa - prod + d
        g = g_num / g_den
        
        # New Stable C and D calculation
        # This form keeps the log argument from crossing the negative real axis
        exp_dt = cmath.exp(d * self.T)
        
        # Note the different log term structure:
        C_val = (self.r * 1j * phi * self.T) + (self.kappa * self.theta / self.sigma**2) * \
                ((self.kappa - prod - d) * self.T - 2 * cmath.log((1 - g * exp_dt) / (1 - g)))
        
        D_val = ((self.kappa - prod - d) / self.sigma**2) * \
                ((1 - exp_dt) / (1 - g * exp_dt))
        
        return cmath.exp(C_val + D_val * self.v0 + 1j * phi * cmath.log(self.S0))

    def price(self, option_type='call'):
        # Integration limits (0 to infinity, approx 100 is usually enough)
        limit = 100 
        
        def integrand1(phi):
            num = cmath.exp(-1j * phi * cmath.log(self.K)) * self.heston_char_func(phi - 1j) / self.heston_char_func(-1j)
            return (num / (1j * phi)).real

        def integrand2(phi):
            num = cmath.exp(-1j * phi * cmath.log(self.K)) * self.heston_char_func(phi)
            return (num / (1j * phi)).real
        
        P1 = 0.5 + (1 / np.pi) * quad(integrand1, 0, limit)[0]
        P2 = 0.5 + (1 / np.pi) * quad(integrand2, 0, limit)[0]
        
        call_price = self.S0 * P1 - self.K * np.exp(-self.r * self.T) * P2
        
        if option_type == 'call':
            return max(call_price, 0.0)
        else:
            # Put-Call Parity
            return max(call_price - self.S0 + self.K * np.exp(-self.r * self.T), 0.0)

# ==========================================
# 3. MOCK DATA ENGINE (Synthesizing WRDS)
# ==========================================
@st.cache_data
def generate_vol_surface():
    """Generates a synthetic Volatility Surface DataFrame similar to WRDS OptionMetrics"""
    strikes = np.linspace(80, 120, 15)
    maturities = np.linspace(0.1, 2.0, 10)
    data = []
    
    for t in maturities:
        for k in strikes:
            # Create a "Smile": Vol is higher far from ATM (100)
            moneyness = np.log(k / 100)
            # Vol Model: Base + Skew * Moneyness + Curvature * Moneyness^2
            iv = 0.20 - 0.1 * moneyness + 0.5 * moneyness**2
            # Add some noise
            iv += np.random.normal(0, 0.005)
            
            data.append({
                "strike": k,
                "maturity": t,
                "implied_volatility": iv,
                "delta": 0.5 # Placeholder
            })
    return pd.DataFrame(data)

# ==========================================
# 4. APP LAYOUT
# ==========================================

# --- GLOBAL SIDEBAR ---
with st.sidebar:
    st.header("Global Market Data")
    S_curr = st.number_input("Spot Price ($)", value=100.0)
    r_curr = st.number_input("Risk-Free Rate", value=0.05)
    sigma_curr = st.number_input("BSM Volatility", value=0.20)
    T_curr = st.number_input("Time to Maturity (Y)", value=1.0)
    st.divider()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["1. Structuring (Payoff)", "2. Pricing Models (Jump)", "3. Vol Surface (WRDS)"])

# ==========================================
# TAB 1: THE STRUCTURER
# ==========================================
with tab1:
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Add Legs")
        inst_type = st.selectbox("Type", ["Stock", "Vanilla Option", "Zero Bond", "Digital Option"])
        side = st.radio("Side", ["Long", "Short"], horizontal=True)
        pos = 1.0 if side == "Long" else -1.0
        
        # Local Volatility Override UI
        use_local_vol = st.checkbox("Override Global Volatility (Smile)")
        local_vol = st.number_input("Local Volatility (σ)", value=0.20, step=0.01) if use_local_vol else None
        
        if inst_type == "Stock":
            if st.button("Add Stock"):
                st.session_state.portfolio.append(Stock(pos, sigma_override=local_vol))
                st.rerun()
                
        elif inst_type == "Vanilla Option":
            otype = st.radio("Option", ["Call", "Put"], horizontal=True)
            k = st.number_input("Strike", value=100.0, step=0.5)
            if st.button("Add Leg"):
                st.session_state.portfolio.append(VanillaOption(k, otype, pos, sigma_override=local_vol))
                st.rerun()
                
        elif inst_type == "Digital Option":
            otype = st.radio("Digi Type", ["Call", "Put"], horizontal=True)
            k = st.number_input("Digi Strike", value=100.0, step=0.5)
            pay = st.number_input("Payout", value=1.0)
            if st.button("Add Digital"):
                st.session_state.portfolio.append(DigitalOption(k, pay, otype, pos, sigma_override=local_vol))
                st.rerun()

        elif inst_type == "Zero Bond":
            face = st.number_input("Face Value", value=100.0)
            if st.button("Add Bond"):
                st.session_state.portfolio.append(ZeroCouponBond(face, pos, sigma_override=local_vol))
                st.rerun()
        
        st.divider()
        st.markdown("**Current Portfolio:**")
        
        if st.session_state.portfolio:
            for i, leg in enumerate(st.session_state.portfolio):
                col_name, col_del = st.columns([4, 1])
                premium = leg.price(S_curr, T_curr, r_curr, sigma_curr)
                # Indicate if local vol is active
                vol_badge = f" (σ={leg.sigma_override:.2f})" if leg.sigma_override else ""
                col_name.text(f"{i+1}. {leg.name()}{vol_badge} | ${abs(premium):.2f}")
                
                if col_del.button("❌", key=f"del_{i}"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()
            
            st.write("") 
            if st.button("Clear All"):
                st.session_state.portfolio = []
                st.rerun()
        else:
            st.info("Empty Portfolio")

    with col_viz:
        st.subheader("Payoff Analysis")
        if st.session_state.portfolio:
            # View Toggle for Absolute vs PnL
            view_mode = st.radio("Chart View", ["Absolute Value", "Net PnL (Cost Adjusted)"], horizontal=True)
            
            s_range = np.linspace(S_curr * 0.5, S_curr * 1.5, 200)
            payoff_total = np.zeros_like(s_range)
            price_total = np.zeros_like(s_range)
            
            port_delta, port_gamma, port_vega, port_theta = 0.0, 0.0, 0.0, 0.0
            initial_cost = 0.0
            
            for leg in st.session_state.portfolio:
                payoff_total += leg.payoff(s_range)
                price_total += leg.price(s_range, T_curr, r_curr, sigma_curr)
                
                # Aggregate Greeks
                port_delta += leg.delta(S_curr, T_curr, r_curr, sigma_curr)
                port_gamma += leg.gamma(S_curr, T_curr, r_curr, sigma_curr)
                port_vega += leg.vega(S_curr, T_curr, r_curr, sigma_curr)
                port_theta += leg.theta(S_curr, T_curr, r_curr, sigma_curr)
                
                # Aggregate Day 0 Cost
                initial_cost += leg.price(S_curr, T_curr, r_curr, sigma_curr)

            # Apply PnL Transformation if selected
            if view_mode == "Net PnL (Cost Adjusted)":
                payoff_total -= initial_cost
                price_total -= initial_cost
                y_axis_title = "Net PnL ($)"
            else:
                y_axis_title = "Absolute Value ($)"

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_range, y=payoff_total, name="Expiry Payoff", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=s_range, y=price_total, name="Current Price (BSM)", line=dict(width=3)))
            
            # Add a zero line if viewing PnL
            if view_mode == "Net PnL (Cost Adjusted)":
                fig.add_hline(y=0, line_color="red", opacity=0.5, line_width=1)
                
            fig.add_vline(x=S_curr, line_dash="dot", annotation_text="Spot")
            fig.update_layout(title="Structure Value", xaxis_title="Spot", yaxis_title=y_axis_title)
            st.plotly_chart(fig, use_container_width=True)
            
            # Value Metric & Greeks Table
            st.metric("Total Portfolio Value (Day 0 Cost)", f"${initial_cost:.2f}")
            
            st.divider()
            st.markdown("#### **Portfolio Greeks**")
            greeks_df = pd.DataFrame({
                "Delta (Δ)": [f"{port_delta:.4f}"],
                "Gamma (Γ)": [f"{port_gamma:.4f}"],
                "Theta (Θ)": [f"{port_theta:.4f}"],
                "Vega (ν)": [f"{port_vega:.4f}"]
            })
            st.table(greeks_df.style.set_properties(**{'text-align': 'center'}))

# ==========================================
# TAB 2: ADVANCED PRICING (Merton & Heston)
# ==========================================
with tab2:
    st.subheader("Advanced Pricing Models")
    st.markdown("Compare standard Black-Scholes against models that handle **Jumps (Merton)** or **Stochastic Volatility (Heston)**.")
    
    # --- MODEL SELECTOR ---
    model_choice = st.radio("Select Pricing Model:", ["Merton Jump Diffusion", "Heston Stochastic Vol"], horizontal=True)
    
    col_params, col_plot = st.columns([1, 2])
    
    # --- DYNAMIC PARAMETERS ---
    with col_params:
        if model_choice == "Merton Jump Diffusion":
            st.info("Configuration: Merton (1976)")
            m_lam = st.slider("Jump Intensity (λ)", 0.0, 5.0, 1.0, help="Jumps per year")
            m_gamma = st.slider("Mean Jump Size (γ)", -0.5, 0.5, -0.1, help="Avg jump size (-0.1 = -10%)")
            m_delta = st.slider("Jump Volatility (δ)", 0.0, 0.5, 0.1, help="Std Dev of jump size")
            
        elif model_choice == "Heston Stochastic Vol":
            st.info("Configuration: Heston (1993)")
            h_v0 = st.slider("Initial Variance (v0)", 0.01, 0.5, 0.04, step=0.01)
            h_kappa = st.slider("Mean Reversion (κ)", 0.1, 5.0, 2.0)
            h_theta = st.slider("Long-Run Variance (θ)", 0.01, 0.5, 0.04)
            h_sigma = st.slider("Vol of Vol (ξ)", 0.1, 1.0, 0.3)
            h_rho = st.slider("Correlation (ρ)", -0.99, 0.99, -0.7, help="Correlation between Stock & Vol")

    # --- PLOTTING ENGINE ---
    with col_plot:
        if not st.session_state.portfolio:
            st.warning("Please add instruments in Tab 1 first.")
        else:
            s_range = np.linspace(S_curr * 0.7, S_curr * 1.3, 50)
            bsm_prices = np.zeros_like(s_range)
            model_prices = np.zeros_like(s_range)
            
            # 1. CALCULATE PRICES
            # We loop through the Spot Range to draw the curve
            for i, s in enumerate(s_range):
                for leg in st.session_state.portfolio:
                    # A. Always calc BSM for baseline
                    bsm_prices[i] += leg.price(s, T_curr, r_curr, sigma_curr)
                    
                    # B. Calc Advanced Model
                    if model_choice == "Merton Jump Diffusion":
                         # Ensure we use the robust function check we fixed earlier
                         model_prices[i] += merton_jump_diffusion_price(leg, s, T_curr, r_curr, sigma_curr, m_lam, m_gamma, m_delta)
                         
                    elif model_choice == "Heston Stochastic Vol":
                        # Heston requires 'VanillaOption' check too
                        if leg.__class__.__name__ == "VanillaOption":
                            # Create Heston Pricer instance for this specific spot price 's'
                            hp = HestonPricer(s, leg.K, T_curr, r_curr, h_v0, h_kappa, h_theta, h_sigma, h_rho)
                            price = hp.price(leg.option_type)
                            model_prices[i] += leg.position * price
                        else:
                            # Fallback to BSM for Digital/Bond if Heston not implemented for them
                            model_prices[i] += leg.price(s, T_curr, r_curr, sigma_curr)

            # 2. DRAW CHART
            fig_adv = go.Figure()
            fig_adv.add_trace(go.Scatter(x=s_range, y=bsm_prices, name="Black-Scholes", line=dict(color='gray', dash='dot')))
            fig_adv.add_trace(go.Scatter(x=s_range, y=model_prices, name=model_choice, line=dict(color='blue', width=3)))
            
            fig_adv.add_vline(x=S_curr, line_dash="dot", annotation_text="Spot")
            fig_adv.update_layout(
                title=f"Price Impact: {model_choice} vs BSM",
                xaxis_title="Spot Price",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_adv, use_container_width=True)

            # 3. METRICS AT CURRENT SPOT
            current_bsm = bsm_prices[np.abs(s_range - S_curr).argmin()]
            current_model = model_prices[np.abs(s_range - S_curr).argmin()]
            diff = current_model - current_bsm
            
            c1, c2 = st.columns(2)
            c1.metric("Black-Scholes Value", f"${current_bsm:.2f}")
            c2.metric(f"{model_choice} Value", f"${current_model:.2f}", delta=f"{diff:.2f}")

# ==========================================
# TAB 3: VOLATILITY SURFACE (WRDS STYLE)
# ==========================================
with tab3:
    st.subheader("Market Reality: Volatility Surface Analysis")
    st.markdown("Visualizing the **Implied Volatility Smile** across Strikes and Maturities (Simulated WRDS OptionMetrics Data).")

    # Generate Mock Data
    df_vol = generate_vol_surface()
    
    # Visualization Type
    viz_type = st.radio("View Type", ["3D Surface", "2D Smile Curve"], horizontal=True)
    
    if viz_type == "3D Surface":
        # Create Pivot Table for 3D Mesh
        vol_pivot = df_vol.pivot(index='maturity', columns='strike', values='implied_volatility')
        
        fig_surf = go.Figure(data=[go.Surface(
            z=vol_pivot.values,
            x=vol_pivot.columns,
            y=vol_pivot.index,
            colorscale='Viridis'
        )])
        fig_surf.update_layout(
            title='Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Maturity (Years)',
                zaxis_title='Implied Vol'
            ),
            width=800, height=600
        )
        st.plotly_chart(fig_surf)
        
    else:
        # 2D Smile
        selected_maturity = st.select_slider("Select Maturity Slice", options=np.unique(df_vol['maturity'].round(2)))
        
        # Filter Data
        # Fuzzy match for float slider
        subset = df_vol[np.isclose(df_vol['maturity'], selected_maturity, atol=0.01)]
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=subset['strike'], 
            y=subset['implied_volatility'], 
            mode='lines+markers',
            name=f'T={selected_maturity}'
        ))
        fig_smile.update_layout(title=f"Volatility Smile (T={selected_maturity:.2f})", xaxis_title="Strike", yaxis_title="Implied Vol")
        st.plotly_chart(fig_smile, use_container_width=True)
    
    with st.expander("View Raw Data (WRDS Format)"):
        st.dataframe(df_vol)
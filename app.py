
"""
VaR Backtesting — Interactive Learning Lab
The Mountain Path Academy
Prof. V. Ravichandran
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from math import comb, log, exp, sqrt, ceil
import io, base64

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VaR Backtesting Lab — The Mountain Path Academy",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand Palette ────────────────────────────────────────────
GOLD = "#FFD700"
BLUE = "#003366"
MID  = "#004d80"
CARD = "#112240"
TXT  = "#e6f1ff"
MUTED = "#8892b0"
GREEN = "#28a745"
RED   = "#dc3545"
LIGHT_BLUE = "#ADD8E6"
AMBER = "#FFC107"
BG_GRAD = "linear-gradient(135deg, #1a2332, #243447, #2a3f5f)"

# ── Global CSS ───────────────────────────────────────────────
st.html(f"""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
:root {{
    --gold: {GOLD}; --blue: {BLUE}; --mid: {MID}; --card: {CARD};
    --txt: {TXT}; --muted: {MUTED}; --green: {GREEN}; --red: {RED};
    --lb: {LIGHT_BLUE}; --amber: {AMBER};
}}
/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1628 0%, #112240 40%, #1a2332 100%) !important;
}}
section[data-testid="stSidebar"] * {{
    color: {TXT} !important; -webkit-text-fill-color: {TXT} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {{
    color: {GOLD} !important; -webkit-text-fill-color: {GOLD} !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {{
    color: {TXT} !important; -webkit-text-fill-color: {TXT} !important;
}}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {{
    background: {CARD} !important; border: 1px solid {MID} !important;
}}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {{
    color: {GOLD} !important; -webkit-text-fill-color: {GOLD} !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: {MID} !important; }}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; }}
.stTabs [data-baseweb="tab"] {{
    background: {CARD}; color: {TXT}; -webkit-text-fill-color: {TXT};
    border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600;
    border: 1px solid {MID}; border-bottom: none;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: {MID}; color: {GOLD} !important;
    -webkit-text-fill-color: {GOLD} !important;
    border-color: {GOLD};
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1rem; }}
/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {CARD}; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid {GOLD};
}}
[data-testid="stMetric"] label {{ color: {MUTED} !important; -webkit-text-fill-color: {MUTED} !important; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {TXT} !important; -webkit-text-fill-color: {TXT} !important;
    font-size: 1.4rem !important; font-weight: 700 !important;
}}
/* ── Expanders ── */
.streamlit-expanderHeader {{ color: {GOLD} !important; -webkit-text-fill-color: {GOLD} !important; font-weight: 600; }}
/* ── Dropdown menu items ── */
[data-baseweb="menu"] {{ background: {CARD} !important; }}
[data-baseweb="menu"] li {{ color: {TXT} !important; -webkit-text-fill-color: {TXT} !important; }}
[data-baseweb="menu"] li:hover {{ background: {MID} !important; }}
/* ── Number input ── */
.stNumberInput input {{ background: {CARD} !important; color: {GOLD} !important;
    -webkit-text-fill-color: {GOLD} !important; border: 1px solid {MID} !important; }}
/* ── Slider ── */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {{
    background: {MID} !important; }}
</style>
""")

# ── Plotly Theme ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(17,34,64,0.85)", plot_bgcolor="rgba(17,34,64,0.5)",
    font=dict(family="Source Sans Pro, sans-serif", color=TXT, size=13),
    title_font=dict(family="Playfair Display, serif", size=18, color=GOLD),
    legend=dict(bgcolor="rgba(17,34,64,0.7)", bordercolor=MID, borderwidth=1, font=dict(color=TXT)),
    xaxis=dict(gridcolor="rgba(136,146,176,0.15)", zerolinecolor=MID),
    yaxis=dict(gridcolor="rgba(136,146,176,0.15)", zerolinecolor=MID),
    margin=dict(l=50, r=30, t=55, b=45),
)

def styled_header(title, subtitle=""):
    html = f"""<div style="user-select:none; padding:18px 24px; background:{CARD};
        border-left:5px solid {GOLD}; border-radius:0 10px 10px 0; margin-bottom:18px;">
        <span style="font-family:'Playfair Display',serif; font-size:1.55rem; font-weight:700;
              color:{GOLD}; -webkit-text-fill-color:{GOLD};">{title}</span>"""
    if subtitle:
        html += f"""<br><span style="font-family:'Source Sans Pro',sans-serif; font-size:0.92rem;
              color:{MUTED}; -webkit-text-fill-color:{MUTED};">{subtitle}</span>"""
    html += "</div>"
    st.html(html)

def info_card(title, value, color=GOLD):
    st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:10px;
        padding:14px 18px; border-left:4px solid {color}; margin-bottom:8px;">
        <span style="font-size:0.78rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};
              font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">{title}</span><br>
        <span style="font-size:1.3rem; font-weight:700; color:{color};
              -webkit-text-fill-color:{color};">{value}</span></div>""")

def zone_badge(zone):
    colors = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}
    icons  = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    c = colors.get(zone, MUTED)
    return f"""<span style="display:inline-block; padding:4px 14px; border-radius:20px;
        background:{c}22; border:2px solid {c}; font-weight:700;
        color:{c}; -webkit-text-fill-color:{c}; font-size:0.95rem;">
        {icons.get(zone,'')} {zone}</span>"""

def footer():
    st.html(f"""<hr style="border-color:{MID}; margin-top:2rem;">
    <div style="user-select:none; text-align:center; padding:12px 0 8px 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700;
              color:{GOLD}; -webkit-text-fill-color:{GOLD};">The Mountain Path Academy</span><br>
        <span style="font-size:0.82rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};">
            Prof. V. Ravichandran &nbsp;|&nbsp;
            Visiting Faculty @ NMIMS Bangalore, BITS Pilani, RV University Bangalore, Goa Institute of Management</span><br>
        <span style="font-size:0.82rem;">
            <a href="https://themountainpathacademy.com" target="_blank"
               style="color:{GOLD}; -webkit-text-fill-color:{GOLD}; text-decoration:none; font-weight:600;">
               🌐 themountainpathacademy.com</a> &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
               style="color:{GOLD}; -webkit-text-fill-color:{GOLD}; text-decoration:none; font-weight:600;">
               💼 LinkedIn</a> &nbsp;|&nbsp;
            <a href="https://github.com/trichyravis" target="_blank"
               style="color:{GOLD}; -webkit-text-fill-color:{GOLD}; text-decoration:none; font-weight:600;">
               💻 GitHub</a></span>
    </div>""")

# ══════════════════════════════════════════════════════════════
#                        SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.html(f"""<div style="user-select:none; text-align:center; padding:10px 0 4px 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:800;
              color:{GOLD}; -webkit-text-fill-color:{GOLD};">🏔️ The Mountain Path</span><br>
        <span style="font-size:0.78rem; color:{LIGHT_BLUE}; -webkit-text-fill-color:{LIGHT_BLUE};
              letter-spacing:1px;">ACADEMY</span></div>""")
    st.divider()

    nav = st.radio("📑 **Navigate**", [
        "🏠 Overview & VaR Methods",
        "📊 Historical VaR Lab",
        "📈 Parametric VaR Lab",
        "🎲 Monte Carlo VaR Lab",
        "🧪 Kupiec POF Test",
        "🔬 Christoffersen Test",
        "🚦 Basel Traffic Light",
        "🏦 Case Study: Bank Assessment",
        "📝 Solved Problems (10)",
        "📚 Educational Reference",
    ], label_visibility="collapsed")

    st.divider()
    st.html(f"""<div style="user-select:none; text-align:center; padding:4px 0;">
        <span style="font-size:0.72rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};">
        For CFA, FRM & MBA Students<br>Financial Risk Management Series</span></div>""")

# ══════════════════════════════════════════════════════════════
#  HELPER: Kupiec LR Calculation
# ══════════════════════════════════════════════════════════════
def kupiec_lr(T, x, p):
    if x == 0 or x == T:
        return 0.0, 1.0  # edge
    pi_hat = x / T
    L0 = x * log(p) + (T - x) * log(1 - p)
    L1 = x * log(pi_hat) + (T - x) * log(1 - pi_hat)
    lr = -2 * (L0 - L1)
    pval = 1 - stats.chi2.cdf(lr, 1)
    return lr, pval

def christoffersen_test(n00, n01, n10, n11):
    T = n00 + n01 + n10 + n11
    x = n01 + n11
    p_hat = x / T if T > 0 else 0
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    # LR UC (Kupiec)
    lr_uc, pval_uc = kupiec_lr(T, x, 0.01)  # we'll recompute with actual p

    # LR IND
    try:
        lr_restricted = 0
        if (1 - p_hat) > 0 and p_hat > 0:
            lr_restricted = (n00 + n10) * log(1 - p_hat) + (n01 + n11) * log(p_hat)
        lr_unrestricted = 0
        if n00 > 0 and (1 - pi01) > 0: lr_unrestricted += n00 * log(1 - pi01)
        if n01 > 0 and pi01 > 0: lr_unrestricted += n01 * log(pi01)
        if n10 > 0 and (1 - pi11) > 0: lr_unrestricted += n10 * log(1 - pi11)
        if n11 > 0 and pi11 > 0: lr_unrestricted += n11 * log(pi11)
        lr_ind = -2 * (lr_restricted - lr_unrestricted)
        pval_ind = 1 - stats.chi2.cdf(lr_ind, 1)
    except:
        lr_ind, pval_ind = 0, 1

    lr_cc = lr_uc + lr_ind
    pval_cc = 1 - stats.chi2.cdf(lr_cc, 2)
    return dict(pi01=pi01, pi11=pi11, p_hat=p_hat, lr_uc=lr_uc, pval_uc=pval_uc,
                lr_ind=lr_ind, pval_ind=pval_ind, lr_cc=lr_cc, pval_cc=pval_cc)

def basel_zone(x):
    if x <= 4: return "GREEN", 3.0, 0.0
    elif x <= 9:
        plus = {5:0.40, 6:0.50, 7:0.65, 8:0.75, 9:0.85}[x]
        return "YELLOW", 3.0 + plus, plus
    else: return "RED", 4.0, 1.0

# ══════════════════════════════════════════════════════════════
#                   PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if nav == "🏠 Overview & VaR Methods":
    styled_header("VaR Backtesting — Interactive Learning Lab",
                   "A Comprehensive Tool for CFA, FRM & MBA Students")

    # Hero banner
    st.html(f"""<div style="user-select:none; background:linear-gradient(135deg,{CARD},{MID});
        border-radius:14px; padding:28px 30px; border:1px solid {GOLD}33; margin-bottom:20px;">
        <span style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:800;
              color:{GOLD}; -webkit-text-fill-color:{GOLD};">What is Value at Risk?</span><br><br>
        <span style="font-size:1.05rem; color:{TXT}; -webkit-text-fill-color:{TXT}; line-height:1.7;">
        VaR estimates the <b>maximum potential loss</b> of a portfolio over a given <b>time horizon</b>
        at a specified <b>confidence level</b>.<br><br>
        <em style="color:{LIGHT_BLUE}; -webkit-text-fill-color:{LIGHT_BLUE};">
        "A 1-day 99% VaR of $500,000 means there is a 1% chance of losing more than $500,000 in a single day."</em>
        </span></div>""")

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Confidence Level", "95% or 99%", GOLD)
        info_card("Time Horizon", "1-day or 10-day", LIGHT_BLUE)
    with c2:
        info_card("Portfolio Value", "Market Value ($)", GREEN)
        info_card("Expected Exceptions (99%, 250d)", "2.5", AMBER)
    with c3:
        info_card("Regulatory Standard", "Basel II / III", MID)
        info_card("Key Measure Shift", "VaR → ES (FRTB)", RED)

    st.markdown("---")
    styled_header("Three VaR Methods", "Compare approaches side-by-side")

    c1, c2, c3 = st.columns(3)
    methods = [
        ("📜 Historical Simulation", "Uses actual past returns to simulate P&L. No distribution assumption.",
         ["✅ Captures fat tails", "✅ No normality assumption", "✅ Easy to explain"],
         ["⚠️ Past may not repeat", "⚠️ Equal weighting", "⚠️ Limited by data window"], LIGHT_BLUE),
        ("📐 Parametric (Var-Cov)", "Assumes returns are normally distributed. VaR = z × σ × V.",
         ["✅ Fastest computation", "✅ Closed-form solution", "✅ Simple inputs"],
         ["⚠️ Underestimates tails", "⚠️ Normal assumption", "⚠️ Poor for non-linear"], GOLD),
        ("🎲 Monte Carlo Simulation", "Generates thousands of random scenarios from a specified model.",
         ["✅ Most flexible", "✅ Handles non-linearity", "✅ Any distribution"],
         ["⚠️ Computationally heavy", "⚠️ Model-dependent", "⚠️ Requires specification"], GREEN),
    ]
    for col, (title, desc, pros, cons, clr) in zip([c1, c2, c3], methods):
        with col:
            pros_html = "".join(f"<div style='color:{GREEN};-webkit-text-fill-color:{GREEN};font-size:0.88rem;'>{p}</div>" for p in pros)
            cons_html = "".join(f"<div style='color:{AMBER};-webkit-text-fill-color:{AMBER};font-size:0.88rem;'>{c}</div>" for c in cons)
            st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:12px;
                padding:18px; border-top:4px solid {clr}; min-height:320px;">
                <span style="font-family:'Playfair Display',serif; font-size:1.1rem;
                      font-weight:700; color:{clr}; -webkit-text-fill-color:{clr};">{title}</span>
                <p style="color:{MUTED}; -webkit-text-fill-color:{MUTED}; font-size:0.88rem;
                   margin:10px 0;">{desc}</p>
                <div style="margin-top:10px;">{pros_html}</div>
                <div style="margin-top:8px;">{cons_html}</div></div>""")

    st.markdown("---")
    styled_header("What is Backtesting?")
    st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:12px; padding:20px;
        border-left:4px solid {RED};">
        <span style="font-size:1rem; color:{TXT}; -webkit-text-fill-color:{TXT}; line-height:1.7;">
        <b style="color:{GOLD}; -webkit-text-fill-color:{GOLD};">Backtesting</b> compares actual portfolio
        losses against VaR predictions. If losses exceed VaR more often than expected, the model
        <b style="color:{RED}; -webkit-text-fill-color:{RED};">underestimates risk</b>.<br><br>
        An <b>exception</b> (breach) occurs on any day where actual loss > VaR. At 99% confidence
        over 250 days, we expect <b style="color:{GOLD}; -webkit-text-fill-color:{GOLD};">2.5 exceptions</b> per year.</span></div>""")

    footer()

# ══════════════════════════════════════════════════════════════
#              PAGE: HISTORICAL VAR LAB
# ══════════════════════════════════════════════════════════════
elif nav == "📊 Historical VaR Lab":
    styled_header("Historical Simulation VaR", "Rank returns and read the percentile — no distribution assumption needed")

    with st.sidebar:
        st.markdown(f"### ⚙️ Parameters")
        port_val = st.number_input("Portfolio Value ($)", value=10_000_000, step=1_000_000, format="%d")
        conf = st.slider("Confidence Level (%)", 90, 99, 99) / 100
        n_days = st.slider("Observation Days", 10, 100, 20)
        seed = st.number_input("Random Seed", value=42, step=1)

    np.random.seed(seed)
    returns = np.random.normal(-0.001, 0.02, n_days)
    pnl = returns * port_val
    sorted_idx = np.argsort(returns)
    sorted_ret = returns[sorted_idx]
    sorted_pnl = pnl[sorted_idx]

    rank = ceil((1 - conf) * n_days)
    var_ret = sorted_ret[rank - 1]
    var_dollar = abs(var_ret) * port_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${port_val:,.0f}")
    c2.metric("Confidence", f"{conf*100:.0f}%")
    c3.metric("VaR Percentile Rank", f"{rank}")
    c4.metric("Historical VaR", f"${var_dollar:,.0f}")

    tab1, tab2 = st.tabs(["📊 Returns & Sorted Distribution", "📋 Data Table"])

    with tab1:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Daily P&L", "Sorted Returns (Worst→Best)"])
        colors = [RED if p < -var_dollar else GREEN for p in pnl]
        fig.add_trace(go.Bar(x=list(range(1, n_days+1)), y=pnl, marker_color=colors,
                             name="Daily P&L", showlegend=False), row=1, col=1)
        fig.add_hline(y=-var_dollar, line_dash="dash", line_color=GOLD, row=1, col=1,
                      annotation_text=f"VaR = -${var_dollar:,.0f}")

        scolors = [RED if i < rank else LIGHT_BLUE for i in range(n_days)]
        fig.add_trace(go.Bar(x=list(range(1, n_days+1)), y=sorted_pnl, marker_color=scolors,
                             name="Sorted P&L", showlegend=False), row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title_text="Historical Simulation VaR")
        fig.update_xaxes(title_text="Day", row=1, col=1)
        fig.update_xaxes(title_text="Rank", row=1, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df = pd.DataFrame({
            "Day": range(1, n_days+1), "Return (%)": [f"{r*100:.3f}" for r in returns],
            "P&L ($)": [f"{p:,.0f}" for p in pnl]
        })
        st.dataframe(df, use_container_width=True, height=400)

    with st.expander("📖 Step-by-Step Calculation"):
        st.markdown(f"""
**Step 1:** Collect {n_days} daily returns.

**Step 2:** Sort returns from worst to best.

**Step 3:** VaR Percentile Rank = ⌈(1 − {conf}) × {n_days}⌉ = **{rank}**

**Step 4:** VaR Return = {rank}th worst return = **{var_ret*100:.4f}%**

**Step 5:** Historical VaR = |{var_ret*100:.4f}%| × ${port_val:,.0f} = **${var_dollar:,.0f}**

> *At {conf*100:.0f}% confidence, the 1-day Historical VaR is ${var_dollar:,.0f}.*
""")
    footer()

# ══════════════════════════════════════════════════════════════
#              PAGE: PARAMETRIC VAR LAB
# ══════════════════════════════════════════════════════════════
elif nav == "📈 Parametric VaR Lab":
    styled_header("Parametric (Variance-Covariance) VaR", "VaR = z × σ × Portfolio Value")

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        port_val = st.number_input("Portfolio Value ($)", value=10_000_000, step=1_000_000, format="%d")
        conf = st.slider("Confidence Level (%)", 90, 99, 99) / 100
        mu = st.number_input("Mean Daily Return (μ %)", value=0.05, step=0.01, format="%.3f") / 100
        sigma = st.number_input("Daily Volatility (σ %)", value=1.80, step=0.10, format="%.2f") / 100

    z = stats.norm.ppf(conf)
    var_ret_no_mean = z * sigma
    var_ret_with_mean = z * sigma - mu
    var_dollar = var_ret_with_mean * port_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Z-Score", f"{z:.4f}")
    c2.metric("VaR Return (no μ)", f"{var_ret_no_mean*100:.3f}%")
    c3.metric("VaR Return (with μ)", f"{var_ret_with_mean*100:.3f}%")
    c4.metric("Parametric VaR", f"${var_dollar:,.0f}")

    tab1, tab2 = st.tabs(["📈 Normal Distribution & VaR", "📊 Volatility Sensitivity"])

    with tab1:
        x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        y_vals = stats.norm.pdf(x_vals, mu, sigma)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy',
                                  fillcolor=f"rgba(173,216,230,0.3)", line=dict(color=LIGHT_BLUE, width=2),
                                  name="Normal PDF"))
        # Shade tail
        tail_x = x_vals[x_vals <= mu - var_ret_with_mean]
        tail_y = stats.norm.pdf(tail_x, mu, sigma)
        fig.add_trace(go.Scatter(x=tail_x, y=tail_y, fill='tozeroy',
                                  fillcolor=f"rgba(220,53,69,0.5)", line=dict(color=RED, width=1),
                                  name=f"Tail ({(1-conf)*100:.0f}%)"))
        fig.add_vline(x=-var_ret_with_mean, line_dash="dash", line_color=GOLD,
                      annotation_text=f"VaR = {var_ret_with_mean*100:.3f}%")
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title_text="Normal Distribution with VaR Threshold",
                          xaxis_title="Daily Return", yaxis_title="Probability Density")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sigmas = np.linspace(0.005, 0.04, 50)
        vars_ = [(z * s - mu) * port_val for s in sigmas]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sigmas*100, y=[v/1e6 for v in vars_],
                                  mode='lines+markers', line=dict(color=GOLD, width=3),
                                  marker=dict(size=4, color=GOLD), name="VaR"))
        fig.add_hline(y=var_dollar/1e6, line_dash="dot", line_color=RED,
                      annotation_text=f"Current: ${var_dollar/1e6:.2f}M")
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title_text="VaR Sensitivity to Daily Volatility",
                          xaxis_title="Daily Volatility σ (%)", yaxis_title="Parametric VaR ($M)")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Z-Score Quick Reference Table"):
        zdf = pd.DataFrame({
            "Confidence": ["90%", "95%", "97.5%", "99%", "99.5%", "99.9%"],
            "Z-Score": [f"{stats.norm.ppf(c):.4f}" for c in [0.90,0.95,0.975,0.99,0.995,0.999]],
            "Meaning": ["1 in 10 days", "1 in 20 days", "1 in 40 days",
                         "1 in 100 days", "1 in 200 days", "1 in 1000 days"]
        })
        st.dataframe(zdf, use_container_width=True, hide_index=True)
    footer()

# ══════════════════════════════════════════════════════════════
#            PAGE: MONTE CARLO VAR LAB
# ══════════════════════════════════════════════════════════════
elif nav == "🎲 Monte Carlo VaR Lab":
    styled_header("Monte Carlo Simulation VaR", "Generate random scenarios → calculate P&L → find VaR percentile")

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        port_val = st.number_input("Portfolio Value ($)", value=10_000_000, step=1_000_000, format="%d")
        conf = st.slider("Confidence Level (%)", 90, 99, 99) / 100
        mu = st.number_input("Mean Daily Return (μ %)", value=0.05, step=0.01, format="%.3f") / 100
        sigma = st.number_input("Daily Volatility (σ %)", value=1.80, step=0.10, format="%.2f") / 100
        n_sims = st.select_slider("Number of Simulations", options=[100,500,1000,5000,10000,50000], value=10000)
        seed = st.number_input("Random Seed", value=42, step=1)

    np.random.seed(seed)
    z_random = np.random.standard_normal(n_sims)
    sim_returns = mu + sigma * z_random
    sim_pnl = sim_returns * port_val
    var_percentile = np.percentile(sim_pnl, (1 - conf) * 100)
    mc_var = abs(var_percentile)

    # Parametric for comparison
    z_par = stats.norm.ppf(conf)
    par_var = (z_par * sigma - mu) * port_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulations", f"{n_sims:,}")
    c2.metric("Monte Carlo VaR", f"${mc_var:,.0f}")
    c3.metric("Parametric VaR", f"${par_var:,.0f}")
    c4.metric("MC vs Parametric", f"{(mc_var/par_var - 1)*100:+.1f}%")

    tab1, tab2 = st.tabs(["📊 Simulated P&L Distribution", "📈 Convergence Analysis"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sim_pnl, nbinsx=100, marker_color=LIGHT_BLUE,
                                    opacity=0.7, name="Simulated P&L"))
        fig.add_vline(x=-mc_var, line_dash="dash", line_color=RED, line_width=2,
                      annotation_text=f"MC VaR = -${mc_var:,.0f}")
        fig.add_vline(x=-par_var, line_dash="dot", line_color=GOLD, line_width=2,
                      annotation_text=f"Parametric VaR = -${par_var:,.0f}")
        fig.update_layout(**PLOTLY_LAYOUT, height=440, title_text="Monte Carlo Simulated P&L Distribution",
                          xaxis_title="P&L ($)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sizes = np.logspace(1.5, np.log10(n_sims), 40).astype(int)
        vars_conv = []
        for s in sizes:
            pnl_sub = sim_pnl[:s]
            vars_conv.append(abs(np.percentile(pnl_sub, (1-conf)*100)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=[v/1e6 for v in vars_conv],
                                  mode='lines', line=dict(color=LIGHT_BLUE, width=2), name="MC VaR"))
        fig.add_hline(y=par_var/1e6, line_dash="dash", line_color=GOLD,
                      annotation_text=f"Parametric VaR = ${par_var/1e6:.3f}M")
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          title_text="MC VaR Convergence (stabilizes with more simulations)",
                          xaxis_title="Number of Simulations", yaxis_title="VaR ($M)",
                          xaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════
#              PAGE: KUPIEC POF TEST
# ══════════════════════════════════════════════════════════════
elif nav == "🧪 Kupiec POF Test":
    styled_header("Kupiec Proportion of Failures (POF) Test",
                   "Tests whether the observed exception count matches the expected rate")

    with st.sidebar:
        st.markdown("### ⚙️ Test Parameters")
        T = st.number_input("Trading Days (T)", value=250, min_value=50, max_value=1000, step=50)
        conf = st.slider("Confidence Level (%)", 90, 99, 99) / 100
        x = st.number_input("Observed Exceptions (x)", value=7, min_value=0, max_value=100, step=1)
        sig_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1,
                                  format_func=lambda v: f"{v*100:.0f}%")

    p = 1 - conf
    expected = p * T
    pi_hat = x / T if T > 0 else 0
    lr, pval = kupiec_lr(T, x, p)
    crit = stats.chi2.ppf(1 - sig_level, 1)
    reject = lr > crit

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Rate (p)", f"{p*100:.1f}%")
    c2.metric("Expected Exceptions", f"{expected:.1f}")
    c3.metric("Observed Rate (π̂)", f"{pi_hat*100:.2f}%")
    c4.metric("Overcount Ratio", f"{x/expected:.1f}×" if expected > 0 else "N/A")

    # Decision banner
    if reject:
        st.html(f"""<div style="user-select:none; background:{RED}22; border:2px solid {RED};
            border-radius:10px; padding:14px 20px; text-align:center;">
            <span style="font-size:1.2rem; font-weight:700; color:{RED};
                  -webkit-text-fill-color:{RED};">❌ REJECT H₀ — VaR Model FAILS the Kupiec Test</span></div>""")
    else:
        st.html(f"""<div style="user-select:none; background:{GREEN}22; border:2px solid {GREEN};
            border-radius:10px; padding:14px 20px; text-align:center;">
            <span style="font-size:1.2rem; font-weight:700; color:{GREEN};
                  -webkit-text-fill-color:{GREEN};">✅ FAIL TO REJECT H₀ — Model PASSES the Kupiec Test</span></div>""")

    tab1, tab2, tab3 = st.tabs(["📊 Step-by-Step Calculation", "📈 LR Visualization", "📋 Sensitivity Table"])

    with tab1:
        L0 = x * log(p) + (T - x) * log(1 - p) if x > 0 else (T - x) * log(1 - p)
        L1 = x * log(pi_hat) + (T - x) * log(1 - pi_hat) if x > 0 and pi_hat > 0 else 0

        steps = pd.DataFrame({
            "Step": ["1. Expected rate (p = 1 - c)", "2. Expected exceptions (p × T)",
                     "3. Observed rate (π̂ = x/T)", "4. Log-Likelihood H₀",
                     "5. Log-Likelihood H₁", "6. LR Statistic = −2(L₀ − L₁)",
                     "7. Critical Value χ²", "8. P-Value", "9. DECISION"],
            "Formula / Value": [f"{p:.4f}", f"{expected:.2f}", f"{pi_hat:.4f}",
                                f"{L0:.4f}", f"{L1:.4f}", f"{lr:.4f}",
                                f"{crit:.4f}", f"{pval:.6f}",
                                "REJECT ❌" if reject else "PASS ✅"],
        })
        st.dataframe(steps, use_container_width=True, hide_index=True, height=380)

    with tab2:
        x_range = np.linspace(0, max(lr * 2, 15), 500)
        y_chi2 = stats.chi2.pdf(x_range, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=y_chi2, fill='tozeroy',
                                  fillcolor=f"rgba(173,216,230,0.2)", line=dict(color=LIGHT_BLUE, width=2),
                                  name="χ²(1) PDF"))
        # Rejection region
        rej_x = x_range[x_range >= crit]
        rej_y = stats.chi2.pdf(rej_x, 1)
        fig.add_trace(go.Scatter(x=rej_x, y=rej_y, fill='tozeroy',
                                  fillcolor=f"rgba(220,53,69,0.4)", line=dict(color=RED, width=1),
                                  name=f"Rejection Region (>{crit:.3f})"))
        fig.add_vline(x=lr, line_dash="dash", line_color=GOLD, line_width=2,
                      annotation_text=f"LR = {lr:.3f}")
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title_text="χ² Distribution with LR Statistic",
                          xaxis_title="LR Statistic", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        rows = []
        for ex in range(0, min(T, 20)):
            lr_i, pv_i = kupiec_lr(T, ex, p)
            crit_i = stats.chi2.ppf(1 - sig_level, 1)
            rows.append({"Exceptions": ex, "Observed Rate": f"{ex/T*100:.2f}%",
                          "LR Statistic": f"{lr_i:.3f}", "P-Value": f"{pv_i:.4f}",
                          "Decision": "❌ REJECT" if lr_i > crit_i else "✅ PASS"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)
    footer()

# ══════════════════════════════════════════════════════════════
#          PAGE: CHRISTOFFERSEN TEST
# ══════════════════════════════════════════════════════════════
elif nav == "🔬 Christoffersen Test":
    styled_header("Christoffersen Conditional Coverage Test",
                   "Tests BOTH exception count AND independence (clustering)")

    with st.sidebar:
        st.markdown("### ⚙️ Transition Matrix")
        n00 = st.number_input("n₀₀ (No→No)", value=238, min_value=0, step=1)
        n01 = st.number_input("n₀₁ (No→Exc)", value=5, min_value=0, step=1)
        n10 = st.number_input("n₁₀ (Exc→No)", value=5, min_value=0, step=1)
        n11 = st.number_input("n₁₁ (Exc→Exc)", value=2, min_value=0, step=1)
        conf_var = st.slider("VaR Confidence (%)", 90, 99, 99) / 100
        sig_level = 0.05

    T_total = n00 + n01 + n10 + n11
    p_var = 1 - conf_var
    x_total = n01 + n11

    # Compute with actual p
    pi_hat_overall = x_total / T_total if T_total > 0 else 0
    lr_uc, pval_uc = kupiec_lr(T_total, x_total, p_var)
    res = christoffersen_test(n00, n01, n10, n11)
    # Override UC with correct p
    res['lr_uc'] = lr_uc
    res['pval_uc'] = pval_uc
    res['lr_cc'] = lr_uc + res['lr_ind']
    res['pval_cc'] = 1 - stats.chi2.cdf(res['lr_cc'], 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days", f"{T_total}")
    c2.metric("Total Exceptions", f"{x_total}")
    c3.metric("π₀₁ (baseline)", f"{res['pi01']*100:.2f}%")
    c4.metric("π₁₁ (after exc)", f"{res['pi11']*100:.2f}%")

    # Clustering indicator
    if res['pi11'] > 0 and res['pi01'] > 0:
        ratio = res['pi11'] / res['pi01']
        if ratio > 3:
            st.html(f"""<div style="user-select:none; background:{RED}22; border:2px solid {RED};
                border-radius:10px; padding:12px 20px; text-align:center;">
                <span style="font-size:1.05rem; font-weight:700; color:{RED};
                      -webkit-text-fill-color:{RED};">⚠️ Clustering Signal: π₁₁ is {ratio:.1f}× higher than π₀₁ — exceptions cluster!</span></div>""")

    tab1, tab2, tab3 = st.tabs(["📊 Transition Matrix", "📋 Three LR Tests", "📈 Visualization"])

    with tab1:
        st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:12px; padding:20px;
            border:1px solid {MID};">
            <table style="width:100%; border-collapse:collapse; color:{TXT}; -webkit-text-fill-color:{TXT}; font-size:1rem;">
            <tr><td></td><td style="text-align:center; padding:8px; font-weight:700; color:{GOLD};
                -webkit-text-fill-color:{GOLD};" colspan="2">Day t</td><td></td></tr>
            <tr><td style="font-weight:700; color:{GOLD}; -webkit-text-fill-color:{GOLD}; padding:8px;">Day t-1</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-weight:600;">No Exc (0)</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-weight:600;">Exc (1)</td>
                <td style="text-align:center; padding:8px; font-weight:600; color:{GOLD}; -webkit-text-fill-color:{GOLD};">Total</td></tr>
            <tr><td style="padding:8px; font-weight:600;">No Exc (0)</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-size:1.2rem;">{n00}</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-size:1.2rem;">{n01}</td>
                <td style="text-align:center; padding:8px; color:{MUTED}; -webkit-text-fill-color:{MUTED};">{n00+n01}</td></tr>
            <tr><td style="padding:8px; font-weight:600;">Exc (1)</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-size:1.2rem;">{n10}</td>
                <td style="text-align:center; padding:8px; border:1px solid {MID}; font-size:1.2rem;
                    background:{RED}22; color:{RED}; -webkit-text-fill-color:{RED}; font-weight:700;">{n11}</td>
                <td style="text-align:center; padding:8px; color:{MUTED}; -webkit-text-fill-color:{MUTED};">{n10+n11}</td></tr>
            <tr><td style="padding:8px; font-weight:600; color:{GOLD}; -webkit-text-fill-color:{GOLD};">Total</td>
                <td style="text-align:center; padding:8px; color:{MUTED}; -webkit-text-fill-color:{MUTED};">{n00+n10}</td>
                <td style="text-align:center; padding:8px; color:{MUTED}; -webkit-text-fill-color:{MUTED};">{n01+n11}</td>
                <td style="text-align:center; padding:8px; font-weight:700; color:{GOLD}; -webkit-text-fill-color:{GOLD};">{T_total}</td></tr>
            </table></div>""")

    with tab2:
        crit1 = stats.chi2.ppf(0.95, 1)
        crit2 = stats.chi2.ppf(0.95, 2)
        tests = [
            ("LR_UC (Kupiec)", res['lr_uc'], crit1, 1, res['pval_uc']),
            ("LR_IND (Independence)", res['lr_ind'], crit1, 1, res['pval_ind']),
            ("LR_CC (Combined)", res['lr_cc'], crit2, 2, res['pval_cc']),
        ]
        rows = []
        for name, lr_v, cr, df, pv in tests:
            decision = "❌ REJECT" if lr_v > cr else "✅ PASS"
            rows.append({"Test": name, "LR Statistic": f"{lr_v:.4f}",
                          f"Critical χ²({df})": f"{cr:.4f}", "P-Value": f"{pv:.6f}",
                          "Decision": decision})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📖 Formula Details"):
            st.latex(r"\pi_{01} = \frac{n_{01}}{n_{00}+n_{01}} \qquad \pi_{11} = \frac{n_{11}}{n_{10}+n_{11}}")
            st.latex(r"LR_{CC} = LR_{UC} + LR_{IND} \sim \chi^2(2)")

    with tab3:
        fig = go.Figure()
        names = ["LR_UC", "LR_IND", "LR_CC"]
        vals = [res['lr_uc'], res['lr_ind'], res['lr_cc']]
        crits = [crit1, crit1, crit2]
        colors_bar = [RED if v > c else GREEN for v, c in zip(vals, crits)]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors_bar, name="LR Statistic", text=[f"{v:.2f}" for v in vals], textposition='auto'))
        fig.add_trace(go.Scatter(x=names, y=crits, mode='markers+lines', marker=dict(size=12, color=GOLD, symbol='diamond'),
                                  line=dict(color=GOLD, dash='dash', width=2), name="Critical Value"))
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title_text="Three Likelihood Ratio Tests",
                          yaxis_title="Statistic Value")
        st.plotly_chart(fig, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════
#            PAGE: BASEL TRAFFIC LIGHT
# ══════════════════════════════════════════════════════════════
elif nav == "🚦 Basel Traffic Light":
    styled_header("Basel Traffic Light Approach",
                   "Regulatory zone classification with capital multiplier impact")

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        exceptions = st.slider("Observed Exceptions", 0, 15, 7)
        var_1d = st.number_input("1-Day VaR ($M)", value=15.0, step=0.5, format="%.1f")

    zone, mult, plus = basel_zone(exceptions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exceptions", f"{exceptions}")
    with c2:
        st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:10px;
            padding:14px 18px; border-left:4px solid {GOLD}; margin-bottom:8px;">
            <span style="font-size:0.78rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};
                  font-weight:600; text-transform:uppercase;">Zone</span><br>
            {zone_badge(zone)}</div>""")
    c3.metric("Multiplier", f"{mult:.2f}×")
    capital = var_1d * sqrt(10) * mult
    capital_green = var_1d * sqrt(10) * 3.0
    c4.metric("Capital Charge", f"${capital:.1f}M")

    tab1, tab2, tab3 = st.tabs(["🚦 Zone Visualization", "📊 Binomial Distribution", "💰 Capital Impact"])

    with tab1:
        fig = go.Figure()
        # Green zone
        fig.add_shape(type="rect", x0=-0.5, x1=4.5, y0=0, y1=1, fillcolor=f"rgba(40,167,69,0.2)",
                      line=dict(width=0))
        # Yellow zone
        fig.add_shape(type="rect", x0=4.5, x1=9.5, y0=0, y1=1, fillcolor=f"rgba(255,193,7,0.2)",
                      line=dict(width=0))
        # Red zone
        fig.add_shape(type="rect", x0=9.5, x1=15.5, y0=0, y1=1, fillcolor=f"rgba(220,53,69,0.2)",
                      line=dict(width=0))
        # Bars - binomial probs
        exc_range = list(range(0, 16))
        probs = [stats.binom.pmf(k, 250, 0.01) for k in exc_range]
        colors_b = [GREEN if k <= 4 else AMBER if k <= 9 else RED for k in exc_range]
        fig.add_trace(go.Bar(x=exc_range, y=probs, marker_color=colors_b,
                              text=[f"{p*100:.1f}%" for p in probs], textposition='auto', name="P(X=x)"))
        fig.add_vline(x=exceptions, line_dash="dash", line_color=GOLD, line_width=3,
                      annotation_text=f"Your: {exceptions}")
        fig.update_layout(**PLOTLY_LAYOUT, height=440,
                          title_text="Basel Zones with Binomial Probability (n=250, p=1%)",
                          xaxis_title="Number of Exceptions", yaxis_title="Probability")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rows = []
        cum = 0
        for k in range(12):
            prob = stats.binom.pmf(k, 250, 0.01)
            cum += prob
            z, m, _ = basel_zone(k)
            rows.append({"Exceptions": k, "P(X=k)": f"{prob*100:.2f}%",
                          "P(X≤k)": f"{cum*100:.2f}%", "Zone": z, "Multiplier": f"{m:.2f}×"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=460)

    with tab3:
        exc_list = list(range(0, 16))
        cap_actual = [var_1d * sqrt(10) * basel_zone(e)[1] for e in exc_list]
        cap_green = [var_1d * sqrt(10) * 3.0] * 16
        fig = go.Figure()
        fig.add_trace(go.Bar(x=exc_list, y=cap_actual, marker_color=[GREEN if e <= 4 else AMBER if e <= 9 else RED for e in exc_list],
                              name="Actual Capital", text=[f"${c:.1f}M" for c in cap_actual], textposition='auto'))
        fig.add_trace(go.Scatter(x=exc_list, y=cap_green, mode='lines', line=dict(color=GOLD, dash='dash', width=2),
                                  name="Green Baseline (3.0×)"))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title_text="Capital Charge by Exception Count",
                          xaxis_title="Exceptions", yaxis_title="Capital ($M)")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        penalty = capital - capital_green
        st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:10px; padding:16px 20px;
            border-left:4px solid {RED if penalty > 0 else GREEN};">
            <span style="font-size:0.85rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};">Capital Penalty vs Green Zone</span><br>
            <span style="font-size:1.5rem; font-weight:700; color:{RED if penalty > 0 else GREEN};
                  -webkit-text-fill-color:{RED if penalty > 0 else GREEN};">
            ${penalty:+.1f}M ({penalty/capital_green*100:+.1f}%)</span></div>""")
    footer()

# ══════════════════════════════════════════════════════════════
#          PAGE: CASE STUDY: BANK ASSESSMENT
# ══════════════════════════════════════════════════════════════
elif nav == "🏦 Case Study: Bank Assessment":
    styled_header("Case Study: Multi-Desk Bank-Wide Assessment",
                   "Meridian Capital Bank — 4 desks, 250 trading days, 99% confidence")

    desks = [
        {"name": "Equities", "asset": "Global equities, ETFs, derivatives", "var": 12.5, "exc": 3},
        {"name": "Fixed Income", "asset": "Govt bonds, credit, IRS", "var": 8.2, "exc": 6},
        {"name": "FX", "asset": "G10 & EM currencies, FX options", "var": 6.8, "exc": 2},
        {"name": "Commodities", "asset": "Energy, metals, agriculture", "var": 9.4, "exc": 11},
    ]

    with st.sidebar:
        st.markdown("### ✏️ Edit Desk Parameters")
        for d in desks:
            d["exc"] = st.number_input(f"{d['name']} Exceptions", value=d["exc"], min_value=0, max_value=50, step=1, key=f"exc_{d['name']}")
            d["var"] = st.number_input(f"{d['name']} 1-Day VaR ($M)", value=d["var"], step=0.5, format="%.1f", key=f"var_{d['name']}")

    rows = []
    total_actual = 0
    total_green = 0
    for d in desks:
        z, m, pf = basel_zone(d["exc"])
        cap = d["var"] * sqrt(10) * m
        cap_g = d["var"] * sqrt(10) * 3.0
        binom_p = 1 - stats.binom.cdf(d["exc"] - 1, 250, 0.01) if d["exc"] > 0 else 1.0
        rows.append({**d, "zone": z, "mult": m, "capital": cap, "capital_green": cap_g,
                      "penalty": cap - cap_g, "binom_p": binom_p})
        total_actual += cap
        total_green += cap_g

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total VaR", f"${sum(d['var'] for d in desks):.1f}M")
    c2.metric("Total Capital", f"${total_actual:.1f}M")
    c3.metric("If All Green", f"${total_green:.1f}M")
    c4.metric("Penalty", f"${total_actual - total_green:.1f}M")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Detailed Table", "📈 Capital Chart"])

    with tab1:
        cols = st.columns(4)
        for col, r in zip(cols, rows):
            with col:
                zone_colors = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}
                border_clr = zone_colors.get(r['zone'], MUTED)
                st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:12px;
                    padding:16px; border-top:4px solid {border_clr}; margin-bottom:8px;">
                    <span style="font-family:'Playfair Display',serif; font-size:1.05rem;
                          font-weight:700; color:{TXT}; -webkit-text-fill-color:{TXT};">{r['name']}</span><br>
                    <span style="font-size:0.8rem; color:{MUTED}; -webkit-text-fill-color:{MUTED};">{r['asset']}</span><br><br>
                    {zone_badge(r['zone'])}<br><br>
                    <span style="color:{TXT}; -webkit-text-fill-color:{TXT}; font-size:0.85rem;">
                    Exceptions: <b>{r['exc']}</b><br>
                    Multiplier: <b>{r['mult']:.2f}×</b><br>
                    Capital: <b>${r['capital']:.1f}M</b><br>
                    P(X≥{r['exc']}): <b>{r['binom_p']*100:.2f}%</b></span></div>""")

    with tab2:
        df = pd.DataFrame([{
            "Desk": r["name"], "Exceptions": r["exc"], "Zone": r["zone"],
            "Multiplier": f"{r['mult']:.2f}×", "Capital ($M)": f"{r['capital']:.1f}",
            "If Green ($M)": f"{r['capital_green']:.1f}",
            "Penalty ($M)": f"{r['penalty']:.1f}",
            "P(X≥x)": f"{r['binom_p']*100:.2f}%"
        } for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        fig = go.Figure()
        names = [r["name"] for r in rows]
        actuals = [r["capital"] for r in rows]
        greens = [r["capital_green"] for r in rows]
        zcolors = [{"GREEN":GREEN,"YELLOW":AMBER,"RED":RED}[r["zone"]] for r in rows]
        fig.add_trace(go.Bar(name="Actual Capital", x=names, y=actuals, marker_color=zcolors,
                              text=[f"${v:.1f}M" for v in actuals], textposition='auto'))
        fig.add_trace(go.Bar(name="If Green (3.0×)", x=names, y=greens,
                              marker_color=f"rgba(40,167,69,0.4)", text=[f"${v:.1f}M" for v in greens], textposition='auto'))
        fig.update_layout(**PLOTLY_LAYOUT, height=440, barmode='group',
                          title_text="Capital Charge: Actual vs All-Green Baseline")
        st.plotly_chart(fig, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════
#          PAGE: SOLVED PROBLEMS
# ══════════════════════════════════════════════════════════════
elif nav == "📝 Solved Problems (10)":
    styled_header("10 Solved Problems — MBA / FRM / CFA Practice",
                   "Fully worked step-by-step solutions with formulas")

    problems = [
        ("Problem 1: Kupiec POF — Model Passes", "99% VaR, 250 days, 5 exceptions"),
        ("Problem 2: Kupiec POF — Model Fails", "95% VaR, 500 days, 40 exceptions"),
        ("Problem 3: Basel Zone & Capital", "8 exceptions, VaR = $15M"),
        ("Problem 4: Christoffersen Independence", "Transition matrix analysis"),
        ("Problem 5: Full Conditional Coverage", "Combined LR_CC test"),
        ("Problem 6: Binomial Probability", "P(X=5) and P(X≥5) for 250 days"),
        ("Problem 7: VaR Scaling 1-Day → 10-Day", "Square-root-of-time rule"),
        ("Problem 8: Multi-Desk Basel", "3 desks with different zones"),
        ("Problem 9: Confidence vs Exceptions", "90/95/99/99.9% comparison"),
        ("Problem 10: Comprehensive Backtest", "Kupiec + Basel + Christoffersen + Recommendation"),
    ]

    prob_idx = st.selectbox("Select Problem", range(10), format_func=lambda i: f"{problems[i][0]}")

    st.html(f"""<div style="user-select:none; background:{CARD}; border-radius:12px; padding:18px 22px;
        border-left:5px solid {GOLD}; margin:12px 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700;
              color:{GOLD}; -webkit-text-fill-color:{GOLD};">{problems[prob_idx][0]}</span><br>
        <span style="color:{MUTED}; -webkit-text-fill-color:{MUTED}; font-size:0.9rem;">{problems[prob_idx][1]}</span></div>""")

    # ── PROBLEM 1 ──
    if prob_idx == 0:
        st.markdown("### Given")
        st.markdown("T = 250, c = 0.99, x = 5, Significance = 5%")
        T, c, x, p = 250, 0.99, 5, 0.01
        pi_hat = x / T
        E_x = p * T
        L0 = x * log(p) + (T-x) * log(1-p)
        L1 = x * log(pi_hat) + (T-x) * log(1-pi_hat)
        lr = -2 * (L0 - L1)
        crit = 3.841
        pval = 1 - stats.chi2.cdf(lr, 1)

        st.markdown("### Step-by-Step Solution")
        steps_data = {
            "Step": ["1. Expected rate p = 1 − c", "2. Expected exceptions E[x] = p × T",
                     "3. Observed rate π̂ = x / T",
                     f"4. L₀ = {x}·ln({p}) + {T-x}·ln({1-p})",
                     f"5. L₁ = {x}·ln({pi_hat:.4f}) + {T-x}·ln({1-pi_hat:.4f})",
                     "6. LR = −2 × (L₀ − L₁)", "7. Compare with χ²(1) critical"],
            "Calculation": [f"1 − 0.99 = {p}", f"0.01 × 250 = {E_x}",
                            f"5 / 250 = {pi_hat:.4f}",
                            f"{L0:.4f}", f"{L1:.4f}",
                            f"−2 × ({L0:.4f} − ({L1:.4f})) = {lr:.4f}",
                            f"{lr:.3f} {'>' if lr > crit else '<'} {crit}"],
            "Result": [f"{p}", f"{E_x}", f"{pi_hat:.4f}", f"{L0:.4f}", f"{L1:.4f}",
                       f"LR = {lr:.4f}", "✅ PASS" if lr < crit else "❌ REJECT"]
        }
        st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)
        st.success(f"**Decision:** LR = {lr:.3f} < {crit} → **FAIL TO REJECT H₀**. Model PASSES. P-value = {pval:.4f}")

    # ── PROBLEM 2 ──
    elif prob_idx == 1:
        st.markdown("### Given")
        st.markdown("T = 500, c = 0.95, x = 40, p = 0.05")
        T, x, p = 500, 40, 0.05
        pi_hat = x / T
        L0 = x*log(p) + (T-x)*log(1-p)
        L1 = x*log(pi_hat) + (T-x)*log(1-pi_hat)
        lr = -2*(L0-L1)
        pval = 1 - stats.chi2.cdf(lr, 1)
        st.markdown("### Solution")
        st.markdown(f"""
| Step | Value |
|------|-------|
| Expected exceptions | 0.05 × 500 = **25** |
| Observed rate π̂ | 40/500 = **{pi_hat}** |
| L₀ | {L0:.4f} |
| L₁ | {L1:.4f} |
| **LR** | **{lr:.4f}** |
| Critical χ²(1) | 3.841 |
| P-value | {pval:.6f} |
""")
        st.error(f"**Decision:** LR = {lr:.3f} > 3.841 → **REJECT H₀**. Model FAILS. 40 exceptions vs 25 expected.")

    # ── PROBLEM 3 ──
    elif prob_idx == 2:
        st.markdown("### Given")
        st.markdown("x = 8 exceptions, VaR₁ = $15M, 99% confidence, 250 days")
        z_name, mult, plus = basel_zone(8)
        cap = 15 * sqrt(10) * mult
        cap_g = 15 * sqrt(10) * 3.0
        st.markdown(f"""### Solution
| Step | Calculation | Result |
|------|-------------|--------|
| 1. Zone | 8 exceptions → 5–9 range | **YELLOW** |
| 2. Plus factor | 8 exc → +0.75 | k = 3.0 + 0.75 = **{mult}×** |
| 3. Capital charge | $15M × √10 × {mult} | **${cap:.1f}M** |
| 4. If Green | $15M × √10 × 3.0 | ${cap_g:.1f}M |
| 5. Penalty | ${cap:.1f} − ${cap_g:.1f} | **${cap-cap_g:.1f}M (+{(cap/cap_g-1)*100:.1f}%)** |
""")
        st.warning(f"The 8 exceptions cost the bank **${cap-cap_g:.1f}M** in additional tied-up capital.")

    # ── PROBLEM 4 ──
    elif prob_idx == 3:
        st.markdown("### Given")
        st.markdown("n₀₀=230, n₀₁=8, n₁₀=7, n₁₁=5, T=250")
        n00, n01, n10, n11 = 230, 8, 7, 5
        pi01 = n01/(n00+n01)
        pi11 = n11/(n10+n11)
        pi_o = (n01+n11)/250
        # Restricted
        Lr = (n00+n10)*log(1-pi_o) + (n01+n11)*log(pi_o)
        Lu = n00*log(1-pi01) + n01*log(pi01) + n10*log(1-pi11) + n11*log(pi11)
        lr_ind = -2*(Lr - Lu)
        pval_ind = 1 - stats.chi2.cdf(lr_ind, 1)
        st.markdown(f"""### Solution
| Quantity | Formula | Result |
|----------|---------|--------|
| π₀₁ | 8/(230+8) | **{pi01:.4f} ({pi01*100:.2f}%)** |
| π₁₁ | 5/(7+5) | **{pi11:.4f} ({pi11*100:.2f}%)** |
| Overall π̂ | 13/250 | **{pi_o:.4f}** |
| L_restricted | | {Lr:.4f} |
| L_unrestricted | | {Lu:.4f} |
| **LR_IND** | −2(L_R − L_U) | **{lr_ind:.4f}** |
| Critical χ²(1) | | 3.841 |
| P-value | | {pval_ind:.6f} |
""")
        st.error(f"**REJECT independence.** π₁₁ = {pi11*100:.1f}% is **{pi11/pi01:.0f}×** higher than π₀₁ = {pi01*100:.1f}%.")

    # ── PROBLEM 5 ──
    elif prob_idx == 4:
        st.markdown("### Given (from Problem 4)")
        st.markdown("T=250, x=13, c=0.99, p=0.01")
        T, x, p = 250, 13, 0.01
        pi_hat = x/T
        lr_uc, pv_uc = kupiec_lr(T, x, p)
        # Independence from P4
        n00, n01, n10, n11 = 230, 8, 7, 5
        pi_o = (n01+n11)/T
        Lr = (n00+n10)*log(1-pi_o) + (n01+n11)*log(pi_o)
        pi01 = n01/(n00+n01); pi11 = n11/(n10+n11)
        Lu = n00*log(1-pi01)+n01*log(pi01)+n10*log(1-pi11)+n11*log(pi11)
        lr_ind = -2*(Lr - Lu)
        lr_cc = lr_uc + lr_ind
        pv_cc = 1 - stats.chi2.cdf(lr_cc, 2)
        st.markdown(f"""### Solution
| Test | LR Statistic | Critical Value | Decision |
|------|-------------|----------------|----------|
| LR_UC (Kupiec) | {lr_uc:.4f} | 3.841 (χ², 1 df) | {'❌ REJECT' if lr_uc > 3.841 else '✅ PASS'} |
| LR_IND (Independence) | {lr_ind:.4f} | 3.841 (χ², 1 df) | {'❌ REJECT' if lr_ind > 3.841 else '✅ PASS'} |
| **LR_CC (Combined)** | **{lr_cc:.4f}** | **5.991 (χ², 2 df)** | **{'❌ REJECT' if lr_cc > 5.991 else '✅ PASS'}** |
""")
        st.error("Model fails **both** count AND independence — structurally flawed.")

    # ── PROBLEM 6 ──
    elif prob_idx == 5:
        st.markdown("### Given: n=250, p=0.01, X ~ Binomial(250, 0.01)")
        p5 = stats.binom.pmf(5, 250, 0.01)
        p_ge5 = 1 - stats.binom.cdf(4, 250, 0.01)
        st.markdown(f"""### Solution
**Part (a): P(X = 5)**

$$P(X=5) = \\binom{{250}}{{5}} (0.01)^5 (0.99)^{{245}} = \\mathbf{{{p5:.4f} = {p5*100:.2f}\\%}}$$

**Part (b): P(X ≥ 5)**

$$P(X \\geq 5) = 1 - P(X \\leq 4) = 1 - {stats.binom.cdf(4,250,0.01):.4f} = \\mathbf{{{p_ge5:.4f} = {p_ge5*100:.2f}\\%}}$$
""")
        st.info(f"There is a **{p_ge5*100:.1f}%** chance of 5+ exceptions even with a **correct** model. This is why Basel's Green zone extends to 4.")

    # ── PROBLEM 7 ──
    elif prob_idx == 6:
        st.markdown("### Given: 1-day 99% VaR = $2.4M, Portfolio = $300M, σ = 1.5%")
        var10 = 2.4 * sqrt(10)
        z99 = stats.norm.ppf(0.99)
        var_check = z99 * 0.015 * 300
        st.markdown(f"""### Solution
**Part (a): Scale to 10-day VaR**

$$\\text{{VaR}}_{{10}} = \\text{{VaR}}_{{1}} \\times \\sqrt{{10}} = \\$2.4M \\times {sqrt(10):.4f} = \\mathbf{{\\${var10:.3f}M}}$$

**Part (b): Verify from first principles**

$$z_{{0.99}} = {z99:.4f}$$
$$\\text{{VaR}}_{{1}} = z \\times \\sigma \\times V = {z99:.4f} \\times 0.015 \\times \\$300M = \\mathbf{{\\${var_check:.2f}M}}$$

> Note: The given VaR of $2.4M implies σ = $2.4M / ({z99:.4f} × $300M) = **{2.4/(z99*300)*100:.3f}%** daily.
""")

    # ── PROBLEM 8 ──
    elif prob_idx == 7:
        st.markdown("### Given: 3 desks over 250 days at 99%")
        desk_data = [("Equities", 3, 8.0), ("Credit", 9, 12.0), ("Rates", 10, 6.5)]
        rows_md = []
        total_a, total_g = 0, 0
        for name, exc, var in desk_data:
            z, m, pf = basel_zone(exc)
            cap = var * sqrt(10) * m
            cap_g = var * sqrt(10) * 3.0
            total_a += cap; total_g += cap_g
            rows_md.append(f"| {name} | {exc} | {z} | {m:.2f}× | ${cap:.1f}M | ${cap_g:.1f}M | ${cap-cap_g:.1f}M |")
        table = "\n".join(rows_md)
        st.markdown(f"""### Solution
| Desk | Exc | Zone | Mult | Capital | If Green | Penalty |
|------|-----|------|------|---------|----------|---------|
{table}
| **Total** | | | | **${total_a:.1f}M** | **${total_g:.1f}M** | **${total_a-total_g:.1f}M** |
""")
        st.warning(f"Total penalty = **${total_a-total_g:.1f}M** ({(total_a/total_g-1)*100:.1f}% above all-Green).")

    # ── PROBLEM 9 ──
    elif prob_idx == 8:
        st.markdown("### Given: 500 trading days, 8 observed exceptions")
        rows9 = []
        for c_val in [0.90, 0.95, 0.99, 0.999]:
            p_val = 1 - c_val
            e_x = p_val * 500
            ratio = 8 / e_x if e_x > 0 else float('inf')
            assessment = "Too conservative" if ratio < 0.5 else "Acceptable" if ratio < 2 else "Excessive" if ratio < 5 else "Seriously flawed"
            rows9.append(f"| {c_val*100:.1f}% | {p_val*100:.1f}% | {e_x:.1f} | {ratio:.2f}× | {assessment} |")
        st.markdown(f"""### Solution
| Confidence | Failure Rate | E[x] | Ratio (8/E[x]) | Assessment |
|------------|-------------|------|----------------|------------|
{chr(10).join(rows9)}
""")
        st.info("The **same 8 exceptions** can be fine at 95% but a severe failure at 99.9% — confidence level choice is critical.")

    # ── PROBLEM 10 ──
    elif prob_idx == 9:
        st.markdown("### Given: Fixed income desk, 99% VaR, 250 days, 6 exceptions, VaR=$18M")
        st.markdown("Transition: n₀₀=239, n₀₁=5, n₁₀=5, n₁₁=1")
        # Kupiec
        T, x, p = 250, 6, 0.01
        lr_uc, pv_uc = kupiec_lr(T, x, p)
        # Basel
        z_name, mult, plus = basel_zone(6)
        cap = 18 * sqrt(10) * mult
        cap_g = 18 * sqrt(10) * 3.0
        # Independence
        n00, n01, n10, n11 = 239, 5, 5, 1
        pi01 = n01/(n00+n01); pi11 = n11/(n10+n11)
        pi_o = (n01+n11)/T
        Lr = (n00+n10)*log(1-pi_o)+(n01+n11)*log(pi_o)
        Lu = n00*log(1-pi01)+n01*log(pi01)+n10*log(1-pi11)+n11*log(pi11)
        lr_ind = -2*(Lr-Lu)
        st.markdown(f"""### Solution
**Part (a): Kupiec** → LR = {lr_uc:.3f} {'<' if lr_uc < 3.841 else '>'} 3.841 → **{'PASS ✅' if lr_uc < 3.841 else 'FAIL ❌'}** (borderline)

**Part (b): Basel** → {z_name} zone, k={mult:.2f}×, Capital = ${cap:.1f}M (penalty = ${cap-cap_g:.1f}M)

**Part (c): Independence** → π₀₁={pi01*100:.2f}%, π₁₁={pi11*100:.2f}%, LR_IND={lr_ind:.3f} {'<' if lr_ind < 3.841 else '>'} 3.841 → **{'PASS ✅' if lr_ind < 3.841 else 'FAIL ❌'}**

**Part (d): Recommendation**
- ✅ Do NOT rebuild (statistical tests pass)
- ⚠️ Recalibrate volatility — shorten lookback window
- 📊 Monitor monthly — one more exception triggers rejection
- 📝 Prepare documentation for regulator
- 💰 Budget for ${cap-cap_g:.1f}M capital penalty
""")
        st.warning("Statistical tests PASS but regulatory zone is YELLOW — model is borderline. Proactive recalibration is advised.")
    footer()

# ══════════════════════════════════════════════════════════════
#          PAGE: EDUCATIONAL REFERENCE
# ══════════════════════════════════════════════════════════════
elif nav == "📚 Educational Reference":
    styled_header("Educational Reference — VaR Backtesting Knowledge Base",
                   "Comprehensive Q&A and concept summaries for exam preparation")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 VaR Fundamentals", "🧪 Backtesting Concepts",
        "📊 Statistical Tests", "🚦 Basel Framework", "🔮 Advanced Topics"
    ])

    with tab1:
        qa_fund = [
            ("What is Value at Risk (VaR)?",
             "VaR estimates the maximum potential loss over a specified time horizon at a given confidence level. A 1-day 99% VaR of $1M means only a 1% probability of losing more than $1M in one day."),
            ("What are the three key parameters?",
             "1) **Confidence Level** (95% or 99%), 2) **Time Horizon** (1-day or 10-day), 3) **Portfolio Value** (current market value)."),
            ("What does VaR NOT tell you?",
             "VaR does NOT measure the magnitude of losses beyond the threshold. If 99% VaR is $1M, the tail loss could be $1.1M or $10M — VaR is silent about the tail. Use **Expected Shortfall (CVaR)** for tail severity."),
            ("How to scale VaR from 1-day to 10-day?",
             "Under the square-root-of-time rule: VaR₁₀ = VaR₁ × √10 ≈ VaR₁ × 3.162. Assumes i.i.d. returns, which may not hold during stress."),
            ("Absolute vs Relative VaR?",
             "**Absolute VaR** = zσV (loss relative to zero). **Relative VaR** = (zσ − μ)V (loss relative to expected return). Difference is the mean μ, often negligible for short horizons."),
            ("How does Historical Simulation work?",
             "Collect historical returns → calculate hypothetical P&L for today's portfolio using each return → sort worst to best → VaR is the loss at the (1−confidence) percentile."),
            ("When is Monte Carlo preferred?",
             "For portfolios with options/derivatives (non-linear payoffs), complex multi-asset portfolios with path-dependent features, or when modelling non-normal distributions. Trade-off: computational cost."),
        ]
        for q, a in qa_fund:
            with st.expander(f"**{q}**"):
                st.markdown(a)

    with tab2:
        qa_bt = [
            ("What is a VaR exception (breach)?",
             "A day where the actual portfolio loss exceeds the VaR estimate. At 99% confidence / 250 days, expect 250 × 1% = **2.5 exceptions** per year."),
            ("Why is backtesting mandatory?",
             "Under **Basel II/III**, banks using Internal Models Approach must backtest daily. Poor results trigger increased capital multipliers, supervisory scrutiny, and potentially model rebuilding."),
            ("What are the two types of backtesting?",
             "1) **Unconditional Coverage** — tests total exception count (Kupiec). 2) **Conditional Coverage** — tests count AND independence (Christoffersen). A model can fail one or both."),
            ("What causes too many exceptions?",
             "Volatility underestimated, correlations break down during stress, fat tails not captured, lookback window too long, model doesn't handle non-linear instruments."),
            ("What causes too few exceptions?",
             "Overly conservative VaR: volatility too high, excessive confidence level, double-counting risk factors. Not penalized by regulators but economically inefficient — too much capital tied up."),
            ("Clean vs Dirty backtesting?",
             "**Clean (hypothetical)**: P&L with frozen portfolio (no new trades) — the regulatory standard. **Dirty (actual)**: P&L including intraday trading — useful internally but muddied by trading effects."),
        ]
        for q, a in qa_bt:
            with st.expander(f"**{q}**"):
                st.markdown(a)

    with tab3:
        qa_tests = [
            ("What is the Kupiec POF test?",
             "A likelihood ratio test (1995) checking if the observed exception rate equals the expected rate. Tests ONLY the count, not timing. LR ~ χ²(1). Critical value at 95%: 3.841."),
            ("Kupiec LR Formula?",
             "LR = −2 × [x·ln(p) + (T−x)·ln(1−p) − x·ln(π̂) − (T−x)·ln(1−π̂)], where T=days, x=exceptions, p=expected rate, π̂=observed rate."),
            ("Main limitation of Kupiec?",
             "Blind to patterns. A model with 5 evenly spread exceptions and 5 consecutive exceptions get the SAME Kupiec result — but the second is clearly worse."),
            ("What does Christoffersen add?",
             "An **independence test** using a transition matrix. Tests P(exc today | exc yesterday) vs P(exc today | no exc yesterday). If they differ significantly, exceptions cluster."),
            ("What is the transition matrix?",
             "A 2×2 table: n₀₀ (no→no), n₀₁ (no→exc), n₁₀ (exc→no), n₁₁ (exc→exc). Key: compare π₀₁ = n₀₁/(n₀₀+n₀₁) vs π₁₁ = n₁₁/(n₁₀+n₁₁). If π₁₁ >> π₀₁, clustering exists."),
            ("Can a model pass Kupiec but fail Christoffersen?",
             "**Yes!** Right number of exceptions but all consecutive → passes Kupiec, fails independence. The dynamic response is broken. Fix: faster volatility updating (lower EWMA λ)."),
            ("Three LR components?",
             "LR_UC (count, χ²(1)) + LR_IND (independence, χ²(1)) = LR_CC (combined, χ²(2)). Critical values: 3.841 for 1 df, 5.991 for 2 df at 95%."),
        ]
        for q, a in qa_tests:
            with st.expander(f"**{q}**"):
                st.markdown(a)

    with tab4:
        qa_basel = [
            ("What is the Basel Traffic Light?",
             "Regulatory framework: **GREEN (0–4 exc)** = 3.0× multiplier, **YELLOW (5–9)** = 3.4×–3.85×, **RED (10+)** = 4.0×. Applied over 250 days at 99% confidence."),
            ("What happens in each zone?",
             "**Green**: No action, base 3.0× multiplier. **Yellow**: Increased multiplier, written explanation required. **Red**: Maximum 4.0× penalty, model must be rebuilt, automatic supervisory action."),
            ("Why 3.0× even in Green?",
             "Accounts for: (1) model risk — all VaR models are approximations, (2) market risks beyond VaR, (3) regulatory conservatism. Applied to max(current VaR, 60-day average VaR)."),
            ("Real-world cost of Red zone?",
             "Multiplier rises from 3.0× to 4.0× = 33% capital increase. For $10M daily VaR: 10-day charge goes from $94.9M to $126.5M — an extra $31.6M tied up."),
            ("Traffic Light vs Statistical tests?",
             "Traffic Light is simpler — just exception count. Doesn't test clustering or statistical significance. Banks typically run all three: Kupiec + Christoffersen for internal validation, Traffic Light for regulatory reporting."),
        ]
        for q, a in qa_basel:
            with st.expander(f"**{q}**"):
                st.markdown(a)

    with tab5:
        qa_adv = [
            ("What is Expected Shortfall (ES / CVaR)?",
             "The average loss in the tail beyond VaR. Always ≥ VaR. Under **Basel III FRTB**, ES at 97.5% is replacing VaR as the primary regulatory risk measure."),
            ("What is Stressed VaR (SVaR)?",
             "Uses a 12-month stress period (e.g., 2008 crisis) for calibration. Required under Basel 2.5/III alongside regular VaR. Capital = max(VaR, 60d avg VaR) × k + max(SVaR, 60d avg SVaR) × k."),
            ("What is FRTB?",
             "Fundamental Review of the Trading Book: (1) ES replaces VaR, (2) desk-level backtesting, (3) failing desks use Standardised Approach, (4) P&L attribution tests added alongside backtesting."),
            ("Common remediation strategies?",
             "1) Shorten lookback window. 2) EWMA weighting. 3) Switch Parametric → Historical/MC. 4) GARCH volatility models. 5) Improve correlation modelling. 6) Add stress scenarios."),
            ("Confidence level vs expected exceptions (250 days)?",
             "90% → 25 exc, 95% → 12.5, 99% → 2.5, 99.9% → 0.25. Higher confidence = fewer exceptions but harder to backtest. 99% balances conservatism with testability."),
            ("How often recalibrate VaR models?",
             "Minimum: daily volatility updates, quarterly parameter recalibration, annual full validation with backtesting, ad-hoc after significant market events."),
        ]
        for q, a in qa_adv:
            with st.expander(f"**{q}**"):
                st.markdown(a)

    st.markdown("---")
    styled_header("📐 Formula Quick Reference")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"\text{Parametric VaR} = z_\alpha \times \sigma \times V")
        st.latex(r"\text{VaR}_{10} = \text{VaR}_1 \times \sqrt{10}")
        st.latex(r"E[x] = (1 - c) \times T")
        st.latex(r"\hat{\pi} = \frac{x}{T}")
    with c2:
        st.latex(r"LR_{POF} = -2\ln\left[\frac{p^x(1-p)^{T-x}}{\hat{\pi}^x(1-\hat{\pi})^{T-x}}\right]")
        st.latex(r"\pi_{01} = \frac{n_{01}}{n_{00}+n_{01}} \quad \pi_{11} = \frac{n_{11}}{n_{10}+n_{11}}")
        st.latex(r"LR_{CC} = LR_{UC} + LR_{IND} \sim \chi^2(2)")
        st.latex(r"\text{Capital} = \text{VaR}_{1d} \times \sqrt{10} \times k")

    st.markdown("---")
    styled_header("📋 Critical Values Quick Reference")
    cv_df = pd.DataFrame({
        "Significance": ["1%", "5%", "10%"],
        "χ²(1 df)": [f"{stats.chi2.ppf(0.99,1):.3f}", f"{stats.chi2.ppf(0.95,1):.3f}", f"{stats.chi2.ppf(0.90,1):.3f}"],
        "χ²(2 df)": [f"{stats.chi2.ppf(0.99,2):.3f}", f"{stats.chi2.ppf(0.95,2):.3f}", f"{stats.chi2.ppf(0.90,2):.3f}"],
        "Z-Score": [f"{stats.norm.ppf(0.99):.4f}", f"{stats.norm.ppf(0.95):.4f}", f"{stats.norm.ppf(0.90):.4f}"],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    footer()

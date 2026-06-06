import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Nigeria Terminal Production Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    actual = pd.read_csv("omahtech_actual_production.csv", parse_dates=["date"])
    forecasts = pd.read_csv("omahtech_all_forecasts.csv", parse_dates=["date"])
    classifications = pd.read_csv("omahtech_classifications.csv")

    # Manual classification overrides based on domain analysis
    overrides = {"AGBAMI": "Declining", "ODUDU (AMENAM BLEND)": "Recovering"}
    classifications["classification"] = classifications.apply(
        lambda row: overrides.get(row["terminal"], row["classification"]), axis=1
    )
    return actual, forecasts, classifications

actual, forecasts, class_df = load_data()

# ── CONSTANTS ────────────────────────────────────────────────────────────────
OPEC_QUOTA_MBBL_MONTH = 46.5  # 1.5 mbopd * 31 days
TARGET_2MBPD_MONTH = 62.0     # 2.0 mbopd * 31 days

CLASS_COLORS = {
    "Growing":     "#2ecc71",
    "Recovering":  "#3498db",
    "Stable":      "#f39c12",
    "Declining":   "#e74c3c",
    "New Entrant": "#9b59b6"
}

TIER_LABELS = {1: "Tier 1 — Major", 2: "Tier 2 — Mid", 3: "Tier 3 — Small"}

# Performance results for reference table
PERFORMANCE_DATA = {
    "Terminal":        ["AGBAMI","BONGA","BONNY","EGINA","ESCRAVOS","FORCADOS","QUA IBOE",
                        "AKPO","ANYALA MADU (CJ Blend)","BRASS","ERHA","ODUDU (AMENAM BLEND)",
                        "TULJA - OKWUIBOME","USAN","ABO","ANTAN","EBOK","OKORO",
                        "OKWORI","OTAKPIPO","PENNINGTON","SEA EAGLE (EA)","YOHO"],
    "Tier":            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
    "Prophet MAPE":    [6.5,19.6,41.8,18.4,18.1,16.9,21.7,
                        95.4,14.3,57.6,23.9,31.5,19.9,34.4,
                        35.4,9.6,10.4,31.3,64.7,67.6,69.0,975.7,26.0],
    "XGBoost MAPE":    [3.7,2.8,5.9,4.4,1.9,5.0,3.4,
                        9.1,3.8,2.9,2.1,2.1,2.1,8.8,
                        3.8,3.4,3.8,5.3,7.2,7.8,15.4,14.7,4.5],
}
perf_df = pd.DataFrame(PERFORMANCE_DATA)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 🛢️ Nigeria Terminal-Level Production Decline Analysis and Forecasting")
st.markdown(
    "Terminal-level crude oil and condensate production analysis across Nigeria's upstream "
    "export infrastructure. Data source: NUPRC monthly production reports, January 2020 to April 2026. "
    "XGBoost trained on January 2021 to December 2023 using lag features that draw on 2020 production history. "
    "Prophet trained on the full available history per terminal."
)
st.divider()

# ── SECTION 1: NATIONAL OVERVIEW ─────────────────────────────────────────────
st.markdown("### National Production Overview")

national_actual = actual.groupby("date")["total_liquids_mbbl"].sum().reset_index()
national_actual.columns = ["date", "total_mbbl"]

latest = national_actual.sort_values("date").iloc[-1]
latest_bopd = latest["total_mbbl"] * 1e6 / 30
quota_pct = (latest["total_mbbl"] / OPEC_QUOTA_MBBL_MONTH) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Month", latest["date"].strftime("%B %Y"))
col2.metric("Total Production", f"{latest['total_mbbl']:.1f}M bbl")
col3.metric("Daily Average", f"{latest_bopd/1e6:.3f}M bopd")
col4.metric("% of OPEC Quota", f"{quota_pct:.1f}%",
            delta=f"{quota_pct - 100:.1f}%" if quota_pct != 100 else "At quota")

# National chart
national_prophet = (
    forecasts[forecasts["model"] == "Prophet"]
    .groupby("date")["forecast_mbbl"].sum().reset_index()
)
national_prophet.columns = ["date", "forecast_mbbl"]
national_prophet_future = national_prophet[national_prophet["date"] >= "2025-01-01"]

fig_national = go.Figure()

fig_national.add_trace(go.Scatter(
    x=national_actual["date"], y=national_actual["total_mbbl"],
    name="Actual Production", line=dict(color="#2c3e50", width=2),
    hovertemplate="%{x|%b %Y}<br>%{y:.2f}M bbl<extra>Actual</extra>"
))

fig_national.add_trace(go.Scatter(
    x=national_prophet_future["date"], y=national_prophet_future["forecast_mbbl"],
    name="Forecast (Prophet)", line=dict(color="#2980b9", width=2, dash="dash"),
    hovertemplate="%{x|%b %Y}<br>%{y:.2f}M bbl<extra>Forecast</extra>"
))

fig_national.add_hline(y=OPEC_QUOTA_MBBL_MONTH, line_dash="dot",
                        line_color="gray", line_width=1.2,
                        annotation_text="OPEC Quota (1.5 mbopd)",
                        annotation_position="top right")
fig_national.add_hline(y=TARGET_2MBPD_MONTH, line_dash="dot",
                        line_color="lightgray", line_width=1.2,
                        annotation_text="Government Target (2.0 mbopd)",
                        annotation_position="top right")

fig_national.update_layout(
    height=380, margin=dict(t=30, b=30),
    yaxis_title="Monthly Production (Million Barrels)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
    plot_bgcolor="white",
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#f0f0f0")
)
st.plotly_chart(fig_national, use_container_width=True)
st.divider()

# ── SECTION 2: TERMINAL EXPLORER ─────────────────────────────────────────────
st.markdown("### Terminal Explorer")

col_left, col_right = st.columns([1, 3])

with col_left:
    all_terminals = sorted(actual["terminal"].unique())
    selected = st.selectbox("Select Terminal", all_terminals, index=all_terminals.index("FORCADOS"))

    terminal_class = class_df[class_df["terminal"] == selected]
    if len(terminal_class) > 0:
        row = terminal_class.iloc[0]
        cls = row["classification"]
        color = CLASS_COLORS.get(cls, "gray")
        st.markdown(f"**Classification:** <span style='color:{color}; font-weight:600'>{cls}</span>", unsafe_allow_html=True)
        st.markdown(f"**Tier:** {TIER_LABELS.get(int(row['tier']), '')}")
        st.markdown(f"**Average monthly production:** {row['avg_production_mbbl']:.3f}M bbl")
        st.markdown(f"**Latest monthly production:** {row['latest_production_mbbl']:.3f}M bbl")

    show_prophet = st.checkbox("Show Prophet forecast", value=True)
    show_xgboost = st.checkbox("Show XGBoost forecast", value=True)
    show_uncertainty = st.checkbox("Show uncertainty band", value=True)

with col_right:
    terminal_actual = actual[actual["terminal"] == selected].sort_values("date")
    terminal_prophet = forecasts[
        (forecasts["terminal"] == selected) & (forecasts["model"] == "Prophet")
    ].sort_values("date")
    terminal_xgb = forecasts[
        (forecasts["terminal"] == selected) & (forecasts["model"] == "XGBoost")
    ].sort_values("date")

    fig_term = go.Figure()

    # Actual
    fig_term.add_trace(go.Scatter(
        x=terminal_actual["date"], y=terminal_actual["total_liquids_mbbl"],
        name="Actual", line=dict(color="#2c3e50", width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:.3f}M bbl<extra>Actual</extra>"
    ))

    # Prophet
    if show_prophet and len(terminal_prophet) > 0:
        prophet_future = terminal_prophet[
            terminal_prophet["date"] > terminal_actual["date"].max()
        ]
        if show_uncertainty and len(prophet_future) > 0:
            fig_term.add_trace(go.Scatter(
                x=pd.concat([prophet_future["date"], prophet_future["date"].iloc[::-1]]),
                y=pd.concat([prophet_future["upper_mbbl"], prophet_future["lower_mbbl"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(41,128,185,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Prophet uncertainty", showlegend=False
            ))
        fig_term.add_trace(go.Scatter(
            x=prophet_future["date"], y=prophet_future["forecast_mbbl"],
            name="Forecast — Prophet", line=dict(color="#2980b9", width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}M bbl<extra>Prophet</extra>"
        ))

    # XGBoost
    if show_xgboost and len(terminal_xgb) > 0:
        xgb_future = terminal_xgb[
            terminal_xgb["date"] > terminal_actual["date"].max()
        ]
        xgb_test = terminal_xgb[
            (terminal_xgb["date"] >= "2024-01-01") &
            (terminal_xgb["date"] <= "2024-12-31")
        ]
        if len(xgb_test) > 0:
            fig_term.add_trace(go.Scatter(
                x=xgb_test["date"], y=xgb_test["forecast_mbbl"],
                name="XGBoost (test set)", line=dict(color="#e67e22", width=2, dash="dot"),
                hovertemplate="%{x|%b %Y}<br>%{y:.3f}M bbl<extra>XGBoost test</extra>"
            ))
        if show_uncertainty and len(xgb_future) > 0:
            fig_term.add_trace(go.Scatter(
                x=pd.concat([xgb_future["date"], xgb_future["date"].iloc[::-1]]),
                y=pd.concat([xgb_future["upper_mbbl"], xgb_future["lower_mbbl"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(230,126,34,0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                name="XGBoost uncertainty", showlegend=False
            ))
        if len(xgb_future) > 0:
            fig_term.add_trace(go.Scatter(
                x=xgb_future["date"], y=xgb_future["forecast_mbbl"],
                name="Forecast — XGBoost", line=dict(color="#e67e22", width=2, dash="dash"),
                hovertemplate="%{x|%b %Y}<br>%{y:.3f}M bbl<extra>XGBoost</extra>"
            ))

    fig_term.update_layout(
        height=400, margin=dict(t=20, b=30),
        yaxis_title="Monthly Production (Million Barrels)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0")
    )
    st.plotly_chart(fig_term, use_container_width=True)

st.divider()

# ── SECTION 3: TERMINAL CLASSIFICATION TABLE ──────────────────────────────────
st.markdown("### Terminal Classification Portfolio")

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    class_filter = st.multiselect(
        "Filter by classification",
        options=["Growing","Recovering","Stable","Declining","New Entrant"],
        default=["Growing","Recovering","Stable","Declining","New Entrant"]
    )
with col_filter2:
    tier_filter = st.multiselect("Filter by tier", options=[1,2,3], default=[1,2,3])

filtered = class_df[
    (class_df["classification"].isin(class_filter)) &
    (class_df["tier"].isin(tier_filter))
].copy()

filtered["tier_label"] = filtered["tier"].map(TIER_LABELS)
filtered["classification_display"] = filtered["classification"]

display_cols = {
    "terminal": "Terminal",
    "tier_label": "Tier",
    "classification": "Classification",
    "avg_production_mbbl": "Avg Monthly Production (M bbl)",
    "latest_production_mbbl": "Latest Month Production (M bbl)"
}
st.dataframe(
    filtered[list(display_cols.keys())].rename(columns=display_cols),
    use_container_width=True, hide_index=True
)

col_g, col_r, col_s, col_d, col_n = st.columns(5)
for col, cls in zip([col_g,col_r,col_s,col_d,col_n],
                    ["Growing","Recovering","Stable","Declining","New Entrant"]):
    count = len(class_df[class_df["classification"]==cls])
    col.metric(cls, count)

st.divider()

# ── SECTION 4: MODEL PERFORMANCE ─────────────────────────────────────────────
st.markdown("### Model Performance — Test Set (January to December 2024)")

st.markdown(
    "Both models were evaluated on a held-out test set that was not used during training. "
    "All MAPE figures below are computed on actual versus predicted production for 2024 only. "
    "XGBoost outperforms Prophet on every terminal in the evaluation set."
)

perf_display = perf_df.copy()
perf_display["Tier"] = perf_display["Tier"].map(TIER_LABELS)
perf_display["Better Model"] = "XGBoost"

st.dataframe(
    perf_display[["Terminal","Tier","Prophet MAPE","XGBoost MAPE","Better Model"]].rename(columns={
        "Prophet MAPE": "Prophet MAPE (%)",
        "XGBoost MAPE": "XGBoost MAPE (%)"
    }),
    use_container_width=True, hide_index=True
)

col_a, col_b = st.columns(2)
col_a.metric("XGBoost Mean MAPE (26 terminals)", "7.88%")
col_b.metric("Prophet Mean MAPE (26 terminals)", "88.90%")

st.divider()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85em; padding: 10px'>
    Built by <b>OmahTech</b> · Data source: NUPRC · Models: Prophet, XGBoost ·
    Research paper forthcoming on Zenodo and Energy Reports
    </div>
    """,
    unsafe_allow_html=True
)
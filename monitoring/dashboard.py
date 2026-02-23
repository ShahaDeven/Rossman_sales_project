import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# CONFIG

st.set_page_config(
    page_title="Rossmann LSTM — Monitoring",
    page_icon="📈",
    layout="wide",
)

# HELPERS
def fetch(endpoint: str):
    try:
        r = requests.get(f"{api_url}{endpoint}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def post_predict(payload: dict):
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown error")
    except Exception as e:
        return None, str(e)


def status_badge(ok: bool):
    return "🟢 Online" if ok else "🔴 Offline"


# PRESET SCENARIOS
PRESETS = {
    "🛒 Busy Friday with Promo": {
        "Store": 1, "DayOfWeek": 5, "Customers": 800, "Open": 1,
        "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0,
        "StoreType": "c", "Assortment": "a", "CompetitionDistance": 1270.0,
        "Promo2": 0, "Year": 2015, "Month": 7, "Day": 10,
        "WeekOfYear": 28, "IsPromoMonth": 0, "IsWeekend": 0,
        "IsHoliday": 0, "IsMonthStart": 0, "IsMonthEnd": 0, "DaysSincePromo": 3,
    },
    "🏖️ Holiday Weekend": {
        "Store": 5, "DayOfWeek": 6, "Customers": 300, "Open": 1,
        "Promo": 0, "StateHoliday": "a", "SchoolHoliday": 1,
        "StoreType": "a", "Assortment": "b", "CompetitionDistance": 3000.0,
        "Promo2": 1, "Year": 2015, "Month": 12, "Day": 26,
        "WeekOfYear": 52, "IsPromoMonth": 1, "IsWeekend": 1,
        "IsHoliday": 1, "IsMonthStart": 0, "IsMonthEnd": 1, "DaysSincePromo": 0,
    },
    "📦 Quiet Monday No Promo": {
        "Store": 3, "DayOfWeek": 1, "Customers": 150, "Open": 1,
        "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0,
        "StoreType": "b", "Assortment": "c", "CompetitionDistance": 500.0,
        "Promo2": 0, "Year": 2015, "Month": 3, "Day": 9,
        "WeekOfYear": 11, "IsPromoMonth": 0, "IsWeekend": 0,
        "IsHoliday": 0, "IsMonthStart": 0, "IsMonthEnd": 0, "DaysSincePromo": 0,
    },
    "🎄 Christmas Eve": {
        "Store": 10, "DayOfWeek": 3, "Customers": 1200, "Open": 1,
        "Promo": 1, "StateHoliday": "c", "SchoolHoliday": 1,
        "StoreType": "d", "Assortment": "a", "CompetitionDistance": 800.0,
        "Promo2": 1, "Year": 2015, "Month": 12, "Day": 24,
        "WeekOfYear": 52, "IsPromoMonth": 1, "IsWeekend": 0,
        "IsHoliday": 1, "IsMonthStart": 0, "IsMonthEnd": 1, "DaysSincePromo": 10,
    },
    "🔒 Store Closed": {
        "Store": 2, "DayOfWeek": 7, "Customers": 0, "Open": 0,
        "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0,
        "StoreType": "a", "Assortment": "a", "CompetitionDistance": 2000.0,
        "Promo2": 0, "Year": 2015, "Month": 6, "Day": 7,
        "WeekOfYear": 23, "IsPromoMonth": 0, "IsWeekend": 1,
        "IsHoliday": 0, "IsMonthStart": 0, "IsMonthEnd": 0, "DaysSincePromo": 0,
    },
}

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (10s)", value=False)
    if st.button("🔄 Refresh Now"):
        st.rerun()
    st.markdown("---")
    st.markdown("**API URL**")
    api_url = st.sidebar.text_input(
        "API URL",
        value="http://localhost:8000"
    )
    st.markdown("**Dashboard**")
    st.caption("Monitoring for Rossmann LSTM Forecasting Service")

# HEADER
st.title("📈 Rossmann Sales Forecasting — Monitoring Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# TABS
tab_monitor, tab_try = st.tabs(["📊 Monitoring", "🧪 Try It Out"])


# TAB 1 — MONITORING
with tab_monitor:

    health     = fetch("/health")
    model_info = fetch("/model-info")
    stats      = fetch("/predictions/stats")
    recent     = fetch("/predictions/recent?limit=200")

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("API Status", status_badge(health is not None))
    with col2:
        version = health.get("model_version", "—") if health else "—"
        st.metric("Model Version", f"v{version}")
    with col3:
        uptime = health.get("uptime_seconds", 0) if health else 0
        st.metric("Uptime", f"{int(uptime//60)}m {int(uptime%60)}s")
    with col4:
        total = stats.get("total_predictions", 0) if stats else 0
        st.metric("Total Predictions", f"{total:,}")
    with col5:
        r2 = model_info.get("metrics", {}).get("test_r2", 0) if model_info else 0
        st.metric("Model R²", f"{r2:.4f}" if r2 else "—")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_lat = stats.get("avg_latency_ms", 0) if stats else 0
        st.metric("Avg Latency", f"{avg_lat:.1f} ms")
    with col2:
        p95_lat = stats.get("p95_latency_ms", 0) if stats else 0
        st.metric("P95 Latency", f"{p95_lat:.1f} ms")
    with col3:
        avg_sales = stats.get("avg_predicted_sales", 0) if stats else 0
        st.metric("Avg Predicted Sales", f"€{avg_sales:,.0f}" if avg_sales else "—")
    with col4:
        mae = model_info.get("metrics", {}).get("test_mae", 0) if model_info else 0
        st.metric("Test MAE", f"€{mae:,.2f}" if mae else "—")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predictions per Hour (last 24h)")
        per_hour = stats.get("predictions_per_hour", []) if stats else []
        if per_hour:
            df_hour = pd.DataFrame(per_hour)
            fig = px.bar(df_hour, x="hour", y="count",
                         color_discrete_sequence=["#4f8ef7"])
            fig.update_layout(
                xaxis_title="Hour", yaxis_title="Predictions",
                margin=dict(l=0, r=0, t=40, b=0), height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions logged yet.")

    with col_right:
        st.subheader("Inference Latency Distribution")
        if recent:
            df_recent = pd.DataFrame(recent)
            if "inference_ms" in df_recent.columns and len(df_recent) > 1:
                fig = px.histogram(df_recent, x="inference_ms", nbins=30,
                                   color_discrete_sequence=["#f7914f"])
                fig.update_layout(
                    xaxis_title="Latency (ms)", yaxis_title="Count",
                    margin=dict(l=0, r=0, t=40, b=0), height=320,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data yet.")
        else:
            st.info("No predictions logged yet.")

    st.divider()

    st.subheader("Predicted Sales Over Time")
    if recent:
        df_recent = pd.DataFrame(recent)
        if "predicted_sales" in df_recent.columns and len(df_recent) > 0:
            df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"])
            df_recent = df_recent.sort_values("timestamp")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_recent["timestamp"], y=df_recent["predicted_sales"],
                mode="lines+markers",
                line=dict(color="#4f8ef7", width=2),
                marker=dict(size=4), name="Predicted Sales",
            ))
            fig.update_layout(
                xaxis_title="Time", yaxis_title="Predicted Sales (€)",
                margin=dict(l=0, r=0, t=40, b=0), height=320,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions logged yet.")
    else:
        st.info("No predictions logged yet.")

    st.divider()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Model Info")
        if model_info:
            st.json({
                "model_name":   model_info.get("model_name"),
                "version":      model_info.get("model_version"),
                "run_id":       model_info.get("run_id", "")[:16] + "...",
                "test_mae":     round(model_info.get("metrics", {}).get("test_mae", 0), 2),
                "test_rmse":    round(model_info.get("metrics", {}).get("test_rmse", 0), 2),
                "test_r2":      round(model_info.get("metrics", {}).get("test_r2", 0), 4),
                "num_features": len(model_info.get("features", [])),
            })
        else:
            st.warning("Could not fetch model info. Is the API running?")

    with col_right:
        st.subheader("Recent Predictions")
        if recent:
            df_table = pd.DataFrame(recent).head(20)
            if not df_table.empty:
                cols = ["timestamp", "store_id", "predicted_sales", "inference_ms", "model_version"]
                cols = [c for c in cols if c in df_table.columns]
                df_table = df_table[cols].copy()
                df_table["predicted_sales"] = df_table["predicted_sales"].apply(lambda x: f"€{x:,.2f}")
                df_table["inference_ms"]    = df_table["inference_ms"].apply(lambda x: f"{x:.1f} ms")
                st.dataframe(df_table, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions logged yet.")


# TAB 2 — TRY IT OUT
with tab_try:
    st.markdown("### 🧪 Try the Model")
    st.markdown(
        "Select a preset scenario or tweak the inputs manually, then hit **Predict** to see the forecast. "
        "No data knowledge required!"
    )
    st.divider()

    preset_name = st.selectbox(
        "📋 Start with a preset scenario:",
        options=list(PRESETS.keys()),
        index=0,
    )
    defaults = PRESETS[preset_name]

    st.divider()

    st.markdown("#### Adjust Inputs")
    st.caption("All fields are pre-filled from the selected preset. Tweak any value and click Predict.")

    with st.expander("📖 What do these fields mean?", expanded=False):
        st.markdown("""
    | Field | Meaning |
    |---|---|
    | **Store ID** | Unique store number (1-1115) |
    | **Store Type** | Store size — a=small, b=large, c=medium, d=extra large |
    | **Assortment** | Product range — a=basic, b=extra, c=extended |
    | **Competition Distance** | Distance in metres to the nearest competitor store |
    | **Promo2** | Whether the store runs a continuous long-term promotion |
    | **Day of Week** | Mon=1 through Sun=7 |
    | **Customers** | Number of customers expected that day |
    | **Store Open** | Whether the store is open (1) or closed (0) |
    | **Promo** | Whether a short-term promotion is active today |
    | **State Holiday** | 0=none, a=public holiday, b=Easter, c=Christmas |
    | **School Holiday** | Whether schools are on holiday |
    | **Days Since Promo** | How many days have passed since the current promo started |
    | **IsWeekend / IsHoliday / IsMonthStart / IsMonthEnd** | Auto-calculated from the inputs above |
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Store Info**")
        store        = st.number_input("Store ID", min_value=1, max_value=1115, value=defaults["Store"])
        store_type   = st.selectbox("Store Type", ["a", "b", "c", "d"],
                                    index=["a","b","c","d"].index(defaults["StoreType"]),
                                    help="a=small, b=large, c=medium, d=extra large")
        assortment   = st.selectbox("Assortment", ["a", "b", "c"],
                                    index=["a","b","c"].index(defaults["Assortment"]),
                                    help="a=basic, b=extra, c=extended")
        comp_dist    = st.number_input("Competition Distance (m)", min_value=0.0,
                                       max_value=100000.0, value=float(defaults["CompetitionDistance"]),
                                       help="Distance to nearest competitor in metres")
        promo2       = st.selectbox("Promo2 (ongoing promo)", [0, 1],
                                    index=defaults["Promo2"],
                                    help="1 = store participates in continuous Promo2")

    with col2:
        st.markdown("**Day Info**")
        day_of_week  = st.selectbox("Day of Week",
                                    options=[1,2,3,4,5,6,7],
                                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1],
                                    index=defaults["DayOfWeek"]-1)
        year         = st.number_input("Year", min_value=2013, max_value=2030, value=defaults["Year"])
        month        = st.selectbox("Month", list(range(1,13)),
                                    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                        "Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
                                    index=defaults["Month"]-1)
        day          = st.number_input("Day", min_value=1, max_value=31, value=defaults["Day"])
        try:
            week_of_year = int(datetime(int(year), int(month), int(day)).strftime("%W"))
            if week_of_year == 0:
                week_of_year = 1
        except:
            week_of_year = defaults["WeekOfYear"]
        st.text_input("Week of Year (auto)", value=str(week_of_year), disabled=True,
                    help="Automatically calculated from Year, Month, Day")

    with col3:
        st.markdown("**Sales Conditions**")
        customers      = st.number_input("Customers", min_value=0, max_value=10000, value=defaults["Customers"],
                                         help="Expected number of customers")
        open_store     = st.selectbox("Store Open", [1, 0],
                                      format_func=lambda x: "Yes" if x == 1 else "No",
                                      index=0 if defaults["Open"]==1 else 1)
        promo          = st.selectbox("Promo Running Today", [1, 0],
                                      format_func=lambda x: "Yes" if x == 1 else "No",
                                      index=0 if defaults["Promo"]==1 else 1)
        state_holiday  = st.selectbox("State Holiday", ["0","a","b","c"],
                                      index=["0","a","b","c"].index(str(defaults["StateHoliday"])),
                                      help="0=none, a=public holiday, b=Easter, c=Christmas")
        school_holiday = st.selectbox("School Holiday", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No",
                                      index=defaults["SchoolHoliday"])

    is_weekend       = 1 if day_of_week in [6, 7] else 0
    is_holiday       = 1 if (school_holiday == 1 or state_holiday != "0") else 0
    is_month_start   = 1 if day <= 3 else 0
    is_month_end     = 1 if day >= 28 else 0
    is_promo_month   = defaults["IsPromoMonth"]
    days_since_promo = st.slider("Days Since Promo Started", min_value=0, max_value=30,
                                  value=defaults["DaysSincePromo"])

    st.caption(f"Auto-calculated → IsWeekend: **{is_weekend}** | IsHoliday: **{is_holiday}** | "
               f"IsMonthStart: **{is_month_start}** | IsMonthEnd: **{is_month_end}**")

    st.divider()

    if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
        payload = {
            "Store":               store,
            "DayOfWeek":           day_of_week,
            "Customers":           customers,
            "Open":                open_store,
            "Promo":               promo,
            "StateHoliday":        state_holiday,
            "SchoolHoliday":       school_holiday,
            "StoreType":           store_type,
            "Assortment":          assortment,
            "CompetitionDistance": comp_dist,
            "Promo2":              promo2,
            "Year":                year,
            "Month":               month,
            "Day":                 day,
            "WeekOfYear":          week_of_year,
            "IsPromoMonth":        is_promo_month,
            "IsWeekend":           is_weekend,
            "IsHoliday":           is_holiday,
            "IsMonthStart":        is_month_start,
            "IsMonthEnd":          is_month_end,
            "DaysSincePromo":      days_since_promo,
        }

        with st.spinner("Calling model..."):
            result, error = post_predict(payload)

        if result:
            st.success("Prediction complete!")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("💰 Predicted Sales", f"€{result['predicted_sales']:,.2f}")
            with r2:
                st.metric("⚡ Inference Latency", f"{result['inference_latency_ms']:.1f} ms")
            with r3:
                st.metric("🤖 Model Version", f"v{result['model_version']}")
        else:
            st.error(f"Prediction failed: {error}")
            st.warning("Make sure the API is running: `uvicorn api.main:app --reload --port 8000`")

# AUTO REFRESH
if auto_refresh:
    time.sleep(10)
    st.rerun()
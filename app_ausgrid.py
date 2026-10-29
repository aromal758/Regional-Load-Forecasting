import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Regional Power Intelligence", page_icon="⚡", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = next((p for p in (BASE_DIR / "data" / "ausgrid_cg_dashboard.npz", BASE_DIR / "ausgrid_cg_dashboard.npz")
                  if p.exists()), BASE_DIR / "data" / "ausgrid_cg_dashboard.npz")
STEPS = 6            # CG-SFDRLF forecasts 1-6 half-hour steps (30 min to 3 h) ahead
ALL = "All 21 regions (total)"


# -----------------------------------------------------------------------------
# Data: Ausgrid Solar Home 2012-13 (272 homes, Sydney) grouped into 21 regions,
# with CG-SFDRLF test-set forecasts exported from the Colab experiment.
# -----------------------------------------------------------------------------
@st.cache_data
def load_store():
    z = np.load(DATA_FILE)
    store = {k: z[k] for k in z.files}
    for k in ("L_kw", "PV_kw", "CS_kw", "muL", "muP"):
        store[k] = store[k].astype(np.float32)
    store["times"] = pd.date_range(str(store["start"]), periods=store["L_kw"].shape[0], freq="30min")
    return store


def region_labels(store):
    labels = [ALL]
    for i, (nh, cap) in enumerate(zip(store["nh"], store["cap"])):
        labels.append(f"Region {i + 1} ({int(nh)} homes, {cap:.0f} kW PV)")
    return labels


def pick(array, region_idx):
    """Select one region (last axis) or sum all regions."""
    return array.sum(-1) if region_idx is None else array[..., region_idx]


def historical_frame(store, region_idx):
    return pd.DataFrame({
        "Time": store["times"],
        "Electricity Load (kW)": pick(store["L_kw"], region_idx),
        "Solar Generation (kW)": pick(store["PV_kw"], region_idx),
    })


def test_blocks(store):
    """Contiguous held-out test periods (2 days each in the 7/1/2 block split)."""
    test = store["split"] == 2
    edges = np.flatnonzero(np.diff(np.r_[0, test.astype(int), 0]))
    blocks = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        t0, t1 = store["times"][a], store["times"][b]
        blocks.append((f"{t0:%d %b %Y} to {t1:%d %b %Y}", a, b))
    return blocks


def test_results(store, region_idx, step=1):
    """Actual vs CG-SFDRLF forecast `step` half-hours ahead, on every test window."""
    target = store["t"] + step
    h = step - 1
    return pd.DataFrame({
        "Time": store["times"][target],
        "Actual Load (kW)": pick(store["L_kw"][target], region_idx),
        "Predicted Load (kW)": pick(store["muL"][:, h, :], region_idx),
        "Actual Solar (kW)": pick(store["PV_kw"][target], region_idx),
        "Predicted Solar (kW)": pick(store["muP"][:, h, :], region_idx),
        "Clear-sky Solar (kW)": pick(store["CS_kw"][target], region_idx),
    })


def in_window(results, window):
    if window is None:
        return results
    _, a, b = window
    times = st.session_state["_times"]
    return results[(results["Time"] >= times[a]) & (results["Time"] <= times[b])]


def with_gaps(results):
    """Insert empty rows between test blocks so lines do not join separate periods."""
    gap = results["Time"].diff() > pd.Timedelta("30min")
    if not gap.any():
        return results
    blanks = results.loc[gap].copy()
    blanks.loc[:, blanks.columns != "Time"] = np.nan
    blanks["Time"] = blanks["Time"] - pd.Timedelta("15min")
    return pd.concat([results, blanks]).sort_values("Time")


def validate_upload(file):
    df = pd.read_csv(file)
    required = ["Time", "Electricity Load (kW)", "Solar Generation (kW)"]
    if not all(c in df.columns for c in required):
        raise ValueError("CSV needs columns: Time, Electricity Load (kW), Solar Generation (kW).")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError("No valid data rows found in the uploaded CSV.")
    if (df[required[1:]] < 0).any().any():
        raise ValueError("Power values cannot be negative.")
    return df[required].sort_values("Time").drop_duplicates("Time").reset_index(drop=True)


def make_chart(data, columns, title):
    fig = go.Figure()
    for column in columns:
        fig.add_trace(go.Scatter(x=data["Time"], y=data[column], name=column, mode="lines"))
    fig.update_layout(
        title=title, template="plotly_dark", height=420,
        hovermode="x unified", xaxis_title="Time", yaxis_title="Power (kW)"
    )
    return fig


try:
    store = load_store()
except Exception as error:
    st.error(f"Unable to load dashboard data ({DATA_FILE.name}): {error}")
    st.stop()
st.session_state["_times"] = store["times"]

st.sidebar.title("⚡ Power Intelligence")
st.sidebar.caption("Prototype • Ausgrid Solar Home data (Sydney, 2012–13) • CG-SFDRLF forecasts")
labels = region_labels(store)
region_label = st.sidebar.selectbox("Region", labels, index=0)
region_idx = None if region_label == ALL else labels.index(region_label) - 1

blocks = test_blocks(store)
pv_by_block = [store["PV_kw"][a:b + 1].sum() for _, a, b in blocks]
default_block = int(np.argmax(pv_by_block))                       # sunniest test period
window_options = ["All test days"] + [b[0] for b in blocks]
window_label = st.sidebar.selectbox("Test period shown in the charts", window_options, index=default_block + 1)
window = None if window_label == "All test days" else blocks[window_options.index(window_label) - 1]

uploaded_file = st.sidebar.file_uploader("Optional: upload a combined CSV dataset", type=["csv"])

try:
    if uploaded_file is not None:
        df = validate_upload(uploaded_file)
        data_source = "Uploaded CSV"
        st.sidebar.success(f"Loaded {len(df):,} valid rows.")
    else:
        df = historical_frame(store, region_idx)
        data_source = "Ausgrid Solar Home dataset (real measurements)"
        st.sidebar.success(f"Loaded {len(df):,} timestamp-aligned historical records.")
except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()

df["Net Load (kW)"] = df["Electricity Load (kW)"] - df["Solar Generation (kW)"]
st.sidebar.download_button(
    "⬇️ Download Current Data", df.to_csv(index=False).encode("utf-8"),
    file_name="regional_power_data.csv", mime="text/csv"
)

st.title("⚡ Regional Electricity Load & Solar Forecasting")
st.caption("CG-SFDRLF Load Forecasting | Solar PV Prediction | Net-Load Analysis")
if uploaded_file is None:
    st.info(
        "REAL DATA: Ausgrid Solar Home dataset — 272 homes with rooftop PV in Sydney, half-hourly, "
        "July 2012 to June 2013, grouped into 21 regions. Forecasts come from the CG-SFDRLF model "
        "on held-out test days that the model never saw during training."
    )
else:
    st.info(f"Using uploaded dataset: {len(df):,} valid records. Uploaded values are inputs, not predictions.")

overview, load_tab, solar_tab, net_tab = st.tabs([
    "🏠 Overview", "⚡ Load Forecasting", "☀️ Solar PV Forecasting", "🔋 Net-Load Analysis"
])

with overview:
    st.subheader(f"Power System Overview — {region_label if uploaded_file is None else 'Uploaded data'}")
    a, b, c, d = st.columns(4)
    a.metric("Peak Load", f"{df['Electricity Load (kW)'].max():.2f} kW")
    b.metric("Peak Solar", f"{df['Solar Generation (kW)'].max():.2f} kW")
    c.metric("Peak Net Load", f"{df['Net Load (kW)'].max():.2f} kW")
    d.metric("Historical Records", f"{len(df):,}")
    fig = make_chart(df, [
        "Electricity Load (kW)", "Solar Generation (kW)", "Net Load (kW)"
    ], "Historical Load, Solar and Net-Load Profile")
    if uploaded_file is None and window is not None:
        # open on two weeks around the selected test period; double-click the chart to see the whole year
        times = store["times"]
        fig.update_xaxes(range=[times[window[1]] - pd.Timedelta(days=6), times[window[2]] + pd.Timedelta(days=6)])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Double-click the chart to zoom out to the full year (July 2012 to June 2013).")
    with st.expander("View Historical Dataset"):
        st.dataframe(df, use_container_width=True)


results_all = test_results(store, region_idx, step=1)
results = with_gaps(in_window(results_all, window))
period_note = "all 72 test days" if window is None else window_label
metric_note = "Metrics use all 72 held-out test days; forecasts are 30 minutes ahead."

with load_tab:
    st.subheader("Electricity Load Analysis")
    st.success("CG-SFDRLF load predictions loaded (Ausgrid test set).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Time"], y=results["Actual Load (kW)"], mode="lines", name="Actual Load"))
    fig.add_trace(go.Scatter(x=results["Time"], y=results["Predicted Load (kW)"], mode="lines", name="CG-SFDRLF Predicted Load"))
    fig.update_layout(title=f"Actual vs Predicted Electricity Load (Test Set: {period_note})", xaxis_title="Time",
                      yaxis_title="Electricity Load (kW)", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)
    actual, pred = results_all["Actual Load (kW)"], results_all["Predicted Load (kW)"]
    c1, c2 = st.columns(2)
    c1.metric("Test MAE", f"{np.mean(np.abs(actual - pred)):.3f} kW")
    c2.metric("Test RMSE", f"{np.sqrt(np.mean((actual - pred) ** 2)):.3f} kW")
    st.info(metric_note)

with solar_tab:
    st.subheader("Solar PV Generation Analysis")
    st.success("CG-SFDRLF solar predictions loaded (Ausgrid test set).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Time"], y=results["Actual Solar (kW)"], mode="lines", name="Actual Solar"))
    fig.add_trace(go.Scatter(x=results["Time"], y=results["Predicted Solar (kW)"], mode="lines", name="CG-SFDRLF Predicted Solar"))
    fig.update_layout(title=f"Actual vs Predicted Solar Generation (Test Set: {period_note})", xaxis_title="Time",
                      yaxis_title="Solar Generation (kW)", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)
    daylight = results_all["Clear-sky Solar (kW)"] > 0.01 * results_all["Clear-sky Solar (kW)"].max()
    actual, pred = results_all.loc[daylight, "Actual Solar (kW)"], results_all.loc[daylight, "Predicted Solar (kW)"]
    c1, c2 = st.columns(2)
    c1.metric("Test MAE (daylight)", f"{np.mean(np.abs(actual - pred)):.3f} kW")
    c2.metric("Test RMSE (daylight)", f"{np.sqrt(np.mean((actual - pred) ** 2)):.3f} kW")
    st.info(metric_note + " Solar errors are measured in daylight hours only.")

with net_tab:
    st.subheader("Predicted Net-Load Analysis")
    try:
        res = results.copy()
        res["Actual Net Load (kW)"] = res["Actual Load (kW)"] - res["Actual Solar (kW)"]
        res["Predicted Net Load (kW)"] = res["Predicted Load (kW)"] - res["Predicted Solar (kW)"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["Time"], y=res["Actual Net Load (kW)"], mode="lines", name="Actual Net Load"))
        fig.add_trace(go.Scatter(x=res["Time"], y=res["Predicted Net Load (kW)"], mode="lines", name="Predicted Net Load"))
        fig.update_layout(title=f"Actual vs Predicted Net Load (Timestamp-Aligned Test Set: {period_note})", xaxis_title="Time",
                          yaxis_title="Net Load (kW)", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        actual = results_all["Actual Load (kW)"] - results_all["Actual Solar (kW)"]
        pred = results_all["Predicted Load (kW)"] - results_all["Predicted Solar (kW)"]
        c1, c2 = st.columns(2)
        c1.metric("Net-Load MAE", f"{np.mean(np.abs(actual - pred)):.3f} kW")
        c2.metric("Net-Load RMSE", f"{np.sqrt(np.mean((actual - pred) ** 2)):.3f} kW")
        st.info("Net load = predicted load − predicted solar on matching timestamps and regions. "
                "Negative values mean local PV exceeds demand (power flows back to the grid). " + metric_note)
    except Exception as error:
        st.error(f"Unable to calculate predicted net load: {error}")

st.divider()
st.header("🔮 Next 3-Hour Power Forecast")
try:
    issue_t = store["t"] if window is None else store["t"][(store["t"] >= window[1] - 1) & (store["t"] <= window[2] - STEPS)]
    issue_labels = [f"{store['times'][t]:%d %b %Y %H:%M}" for t in issue_t]
    default_issue = int(np.argmax([store["times"][t].hour == 12 and store["times"][t].minute == 0 for t in issue_t]))
    issue_label = st.select_slider("Forecast issued at", options=issue_labels, value=issue_labels[default_issue])
    w = int(np.flatnonzero(store["t"] == issue_t[issue_labels.index(issue_label)])[0])
    t0 = int(store["t"][w])
    forecast = pd.DataFrame({
        "Time": store["times"][t0 + 1:t0 + STEPS + 1],
        "Forecast Step": [f"+{30 * (k + 1)} min" for k in range(STEPS)],
        "Predicted Load (kW)": pick(store["muL"][w], region_idx),
        "Predicted Solar (kW)": pick(store["muP"][w], region_idx),
    })
    if forecast.empty:
        raise ValueError("No matching forecast records were found.")
    forecast["Predicted Net Load (kW)"] = forecast["Predicted Load (kW)"] - forecast["Predicted Solar (kW)"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Forecast Load", f"{forecast['Predicted Load (kW)'].mean():.2f} kW")
    c2.metric("Peak Forecast Solar", f"{forecast['Predicted Solar (kW)'].max():.2f} kW")
    c3.metric("Peak Forecast Net Load", f"{forecast['Predicted Net Load (kW)'].max():.2f} kW")
    fig = go.Figure()
    for column in ["Predicted Load (kW)", "Predicted Solar (kW)", "Predicted Net Load (kW)"]:
        fig.add_trace(go.Scatter(x=forecast["Time"], y=forecast[column], mode="lines+markers", name=column))
    fig.update_layout(title="Predicted Load, Solar and Net Load", xaxis_title="Forecast Time",
                      yaxis_title="Power (kW)", template="plotly_dark", height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View 3-Hour Forecast Table"):
        st.dataframe(forecast, use_container_width=True)
    st.download_button("⬇️ Download Future Forecast", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="future_power_forecast.csv", mime="text/csv")
    st.info("CG-SFDRLF forecasts 30 minutes to 3 hours ahead, so this outlook covers the next six half-hour steps "
            "from the selected time in the held-out test period.")
except Exception as error:
    st.error(f"Unable to display future forecasts: {error}")

st.caption(f"Data source: {data_source} | Forecasts: CG-SFDRLF (seed 0) on held-out Ausgrid test days.")

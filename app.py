
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from strategy import add_indicators, make_signal, make_plan, market_regime, signal_quality

st.set_page_config(
    page_title="Finex Live Candle Assistant V8",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TD_BASE = "https://api.twelvedata.com"
FINEX_WEB_URL = "https://finex.co.id/trading/platform-web-mt5"

st.markdown("""
<style>
.block-container{padding-top:.7rem;padding-bottom:2rem;max-width:1120px}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);padding:10px;border-radius:14px}
.hero{font-size:2.8rem;font-weight:850;line-height:1}
.buy{color:#16a34a}.sell{color:#dc2626}.wait{color:#ca8a04}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;background:#64748b22;margin:3px 5px 3px 0}
</style>
""", unsafe_allow_html=True)

CATALOG = {
    "EURUSD":{"name":"Euro / Dolar AS","td":"EUR/USD","contract":100000,"tick":0.00001,"spread":0.00010},
    "GBPUSD":{"name":"Poundsterling / Dolar AS","td":"GBP/USD","contract":100000,"tick":0.00001,"spread":0.00012},
    "AUDUSD":{"name":"Dolar Australia / Dolar AS","td":"AUD/USD","contract":100000,"tick":0.00001,"spread":0.00007},
    "USDJPY":{"name":"Dolar AS / Yen Jepang","td":"USD/JPY","contract":100000,"tick":0.001,"spread":0.012},
    "USDCHF":{"name":"Dolar AS / Franc Swiss","td":"USD/CHF","contract":100000,"tick":0.00001,"spread":0.00012},
    "AUDCAD":{"name":"Dolar Australia / Dolar Kanada","td":"AUD/CAD","contract":100000,"tick":0.00001,"spread":0.00014},
    "AUDCHF":{"name":"Dolar Australia / Franc Swiss","td":"AUD/CHF","contract":100000,"tick":0.00001,"spread":0.00007},
    "AUDJPY":{"name":"Dolar Australia / Yen Jepang","td":"AUD/JPY","contract":100000,"tick":0.001,"spread":0.011},
    "AUDNZD":{"name":"Dolar Australia / Dolar Selandia Baru","td":"AUD/NZD","contract":100000,"tick":0.00001,"spread":0.00012},
    "CADJPY":{"name":"Dolar Kanada / Yen Jepang","td":"CAD/JPY","contract":100000,"tick":0.001,"spread":0.013},
    "CHFJPY":{"name":"Franc Swiss / Yen Jepang","td":"CHF/JPY","contract":100000,"tick":0.001,"spread":0.015},
    "EURAUD":{"name":"Euro / Dolar Australia","td":"EUR/AUD","contract":100000,"tick":0.00001,"spread":0.00011},
    "XAUUSD":{"name":"Gold / Dolar AS","td":"XAU/USD","contract":100,"tick":0.01,"spread":0.30},
    "XAGUSD":{"name":"Silver / Dolar AS","td":"XAG/USD","contract":5000,"tick":0.001,"spread":0.03},
    "AAPL":{"name":"Apple Inc.","td":"AAPL","contract":1,"tick":0.01,"spread":0.10},
    "MSFT":{"name":"Microsoft","td":"MSFT","contract":1,"tick":0.01,"spread":0.10},
    "NVDA":{"name":"NVIDIA","td":"NVDA","contract":1,"tick":0.01,"spread":0.10},
    "TSLA":{"name":"Tesla","td":"TSLA","contract":1,"tick":0.01,"spread":0.10},
    "AMZN":{"name":"Amazon","td":"AMZN","contract":1,"tick":0.01,"spread":0.10},
    "META":{"name":"Meta Platforms","td":"META","contract":1,"tick":0.01,"spread":0.10},
    "GOOG":{"name":"Alphabet / Google","td":"GOOG","contract":1,"tick":0.01,"spread":0.10},
}

INTERVALS = {
    "1 menit":{"td":"1min","seconds":60},
    "5 menit":{"td":"5min","seconds":300},
    "15 menit":{"td":"15min","seconds":900},
    "30 menit":{"td":"30min","seconds":1800},
    "1 jam":{"td":"1h","seconds":3600},
    "1 hari":{"td":"1day","seconds":86400},
}

def secret(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)

API_KEY = secret("TWELVE_DATA_API_KEY","")

def api_get(endpoint, params, timeout=12):
    r = requests.get(f"{TD_BASE}/{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5, show_spinner=False)
def get_price(symbol, api_key):
    try:
        j = api_get("price", {"symbol":symbol,"apikey":api_key})
        if "price" not in j:
            return None, str(j.get("message", j))
        return float(j["price"]), ""
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=30, show_spinner=False)
def get_candles(symbol, interval, api_key):
    try:
        j = api_get("time_series", {
            "symbol":symbol,
            "interval":interval,
            "outputsize":500,
            "order":"ASC",
            "format":"JSON",
            "apikey":api_key,
        })
        vals = j.get("values", [])
        if not vals:
            return pd.DataFrame(), str(j.get("message","No values"))
        df = pd.DataFrame(vals)
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["time"] = pd.to_datetime(df["datetime"])
        df = df[["time","open","high","low","close","volume"]].dropna(subset=["open","high","low","close"])
        return df, ""
    except Exception as e:
        return pd.DataFrame(), str(e)

def interval_floor(ts, seconds):
    if seconds >= 86400:
        return ts.normalize()
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % seconds)
    return pd.Timestamp(floored, unit="s")

def patch_live_candle(df, live_price, seconds, symbol):
    """
    Make the active candle visibly move with each live-price refresh.
    The API time-series may lag or only update after candle boundaries.
    We preserve historical candles and maintain a session-local active OHLC candle.
    """
    now = pd.Timestamp.now().tz_localize(None)
    bucket = interval_floor(now, seconds)
    state_key = f"live_candle_{symbol}_{seconds}"

    # Pick the most recent API close as a fallback open.
    last = df.iloc[-1].copy()
    last_time = pd.Timestamp(last["time"]).tz_localize(None) if getattr(pd.Timestamp(last["time"]), "tzinfo", None) else pd.Timestamp(last["time"])
    latest_open = float(last["close"])

    # If API already has a row in current bucket, use it as seed.
    same_bucket = interval_floor(last_time, seconds) == bucket
    if same_bucket:
        latest_open = float(last["open"])

    state = st.session_state.get(state_key)
    if not state or pd.Timestamp(state["time"]) != bucket:
        state = {
            "time": bucket,
            "open": latest_open,
            "high": max(latest_open, live_price),
            "low": min(latest_open, live_price),
            "close": live_price,
            "volume": 0.0,
        }
    else:
        state["high"] = max(float(state["high"]), live_price)
        state["low"] = min(float(state["low"]), live_price)
        state["close"] = live_price

    st.session_state[state_key] = state

    out = df.copy()
    # Remove any API row belonging to the same active bucket, then append our live candle.
    keep = []
    for _, r in out.iterrows():
        rt = pd.Timestamp(r["time"])
        if getattr(rt, "tzinfo", None):
            rt = rt.tz_localize(None)
        keep.append(interval_floor(rt, seconds) != bucket)
    out = out[pd.Series(keep, index=out.index)].copy()

    new_row = pd.DataFrame([state])
    out = pd.concat([out, new_row], ignore_index=True)
    out = out.sort_values("time").reset_index(drop=True)
    return out, state

st.title("📈 Finex Live Candle Assistant V8")
st.caption("Candle aktif sekarang bergerak mengikuti live market price API pada setiap refresh.")

with st.expander("⚙️ API", expanded=not bool(API_KEY)):
    api_key = st.text_input("Twelve Data API Key", value=API_KEY, type="password")
    st.caption("Simpan sebagai TWELVE_DATA_API_KEY di Streamlit Secrets agar otomatis.")

if not api_key:
    st.warning("Masukkan API key terlebih dahulu.")
    st.stop()

c1,c2,c3 = st.columns([1.8,1,1])
with c1:
    code = st.selectbox("Instrumen", list(CATALOG), format_func=lambda x:f"{x} — {CATALOG[x]['name']}")
with c2:
    tf = st.selectbox("Timeframe", list(INTERVALS), index=2)
with c3:
    refresh_mode = st.selectbox("Refresh candle", ["5 detik","10 detik","20 detik","60 detik"], index=1)

refresh_sec = int(refresh_mode.split()[0])
st_autorefresh(interval=refresh_sec*1000, key="autorefresh_v8")

spec = CATALOG[code]
iv = INTERVALS[tf]

live_price, perr = get_price(spec["td"], api_key)
df, cerr = get_candles(spec["td"], iv["td"], api_key)

if live_price is None:
    st.error(f"Live price gagal: {perr}")
    st.stop()
if df.empty:
    st.error(f"Candle gagal: {cerr}")
    st.stop()

# Make the latest candle move on every refresh.
df, active = patch_live_candle(df, live_price, iv["seconds"], code)

# Align last chart close exactly to live price.
last_close = float(df.iloc[-1]["close"])
if last_close and live_price:
    factor = live_price / last_close
else:
    factor = 1.0

df = add_indicators(df)
sig = make_signal(df)
plan = make_plan(df, sig, atr_mult=1.5, rr=2.0)
regime = market_regime(df)
quality = signal_quality(df, sig)

spread = float(spec["spread"])
bid = live_price - spread/2
ask = live_price + spread/2

st.markdown(
    '<span class="badge">Live price refresh</span>'
    '<span class="badge">Active candle patched</span>'
    f'<span class="badge">{tf}</span>',
    unsafe_allow_html=True
)

m1,m2,m3,m4 = st.columns(4)
m1.metric("Live market", f"{live_price:,.5f}")
m2.metric("Auto BID", f"{bid:,.5f}")
m3.metric("Auto ASK", f"{ask:,.5f}")
m4.metric("Refresh", f"{refresh_sec} detik")

st.subheader("Candle aktif")
a1,a2,a3,a4 = st.columns(4)
a1.metric("Open", f"{active['open']:,.5f}")
a2.metric("High", f"{active['high']:,.5f}")
a3.metric("Low", f"{active['low']:,.5f}")
a4.metric("Close / Live", f"{active['close']:,.5f}")

st.caption(
    f"Candle aktif dimulai {pd.Timestamp(active['time']).strftime('%Y-%m-%d %H:%M:%S')} "
    f"dan diperbarui dari live price pada {datetime.now().strftime('%H:%M:%S')}."
)

st.subheader("Sinyal")
s1,s2,s3,s4 = st.columns(4)
s1.metric("Sinyal", sig["signal"])
s2.metric("Skor", f"{sig['score']}/5")
s3.metric("Quality", f"{quality['score']}/100")
s4.metric("Regime", regime["label"])

klass={"BUY":"buy","SELL":"sell","WAIT":"wait"}[sig["signal"]]
st.markdown(f'<div class="hero {klass}">{sig["signal"]}</div>', unsafe_allow_html=True)

p1,p2,p3 = st.columns(3)
fmt=lambda x:"-" if not np.isfinite(x) else f"{x:,.5f}"
p1.metric("Entry indikatif", fmt(plan["entry"]))
p2.metric("Stop Loss", fmt(plan["sl"]))
p3.metric("Take Profit", fmt(plan["tp"]))

plot = df.tail(180)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.03, row_heights=[.62,.18,.20])
fig.add_trace(go.Candlestick(
    x=plot.time, open=plot.open, high=plot.high, low=plot.low, close=plot.close, name=code
), row=1, col=1)
fig.add_trace(go.Scatter(x=plot.time, y=plot.ema_fast, name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot.time, y=plot.ema_slow, name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot.time, y=plot.rsi, name="RSI"), row=2, col=1)
fig.add_hline(y=70, row=2, col=1)
fig.add_hline(y=30, row=2, col=1)
fig.add_trace(go.Bar(x=plot.time, y=plot.macd_hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot.time, y=plot.macd, name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot.time, y=plot.macd_signal, name="MACD Signal"), row=3, col=1)
fig.update_layout(
    height=680,
    xaxis_rangeslider_visible=False,
    legend_orientation="h",
    margin=dict(l=15,r=15,t=15,b=15),
    uirevision=f"{code}_{tf}",  # Preserve zoom/pan while data updates.
)
st.plotly_chart(fig, use_container_width=True, key=f"chart_{code}_{tf}")

st.subheader("Decision Gate")
checks = [
    ("Live price tersedia", live_price is not None),
    ("Candle aktif mengikuti live price", abs(float(active["close"])-live_price) <= max(spec["tick"]*2, 1e-9)),
    ("Sinyal BUY/SELL", sig["signal"]!="WAIT"),
    ("Skor ≥4/5", sig["score"]>=4),
    ("Quality ≥70/100", quality["score"]>=70),
    ("Pasar tidak sideways", regime["tradeable"]),
]
for label, ok in checks:
    st.write(("✅" if ok else "❌"), label)

if all(ok for _,ok in checks):
    st.success("Setup memenuhi filter aplikasi. Cek quote broker sebelum order.")
else:
    st.info("Belum memenuhi seluruh filter.")

with st.expander("Kenapa candle V7 tampak diam?", expanded=False):
    st.write(
        "Endpoint time-series dari provider dapat memperbarui candle lebih lambat daripada endpoint live price. "
        "V8 mengatasi ini dengan membangun candle aktif secara lokal: Open dipertahankan selama timeframe yang sama, "
        "High/Low diperbarui ketika live price membuat ekstrem baru, dan Close selalu mengikuti live price terbaru."
    )

st.link_button("🔗 BUKA FINEX WEB MT5", FINEX_WEB_URL, use_container_width=True)

st.warning(
    "Candle aktif V8 mengikuti market price API pihak ketiga, bukan tick-by-tick feed resmi Finex. "
    "Auto BID/ASK tetap estimasi dari market mid + spread acuan. Quote broker dapat berbeda."
)

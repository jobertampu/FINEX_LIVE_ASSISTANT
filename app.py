
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
    page_title="Finex Live API Assistant V6",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FINEX_WEB_URL = "https://finex.co.id/trading/platform-web-mt5"
TD_BASE = "https://api.twelvedata.com"

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
    "EURUSD":{"name":"Euro / Dolar AS","td":"EUR/USD","contract":100000,"tick":0.00001},
    "GBPUSD":{"name":"Poundsterling / Dolar AS","td":"GBP/USD","contract":100000,"tick":0.00001},
    "AUDUSD":{"name":"Dolar Australia / Dolar AS","td":"AUD/USD","contract":100000,"tick":0.00001},
    "USDJPY":{"name":"Dolar AS / Yen Jepang","td":"USD/JPY","contract":100000,"tick":0.001},
    "USDCHF":{"name":"Dolar AS / Franc Swiss","td":"USD/CHF","contract":100000,"tick":0.00001},
    "AUDCAD":{"name":"Dolar Australia / Dolar Kanada","td":"AUD/CAD","contract":100000,"tick":0.00001},
    "AUDCHF":{"name":"Dolar Australia / Franc Swiss","td":"AUD/CHF","contract":100000,"tick":0.00001},
    "AUDJPY":{"name":"Dolar Australia / Yen Jepang","td":"AUD/JPY","contract":100000,"tick":0.001},
    "AUDNZD":{"name":"Dolar Australia / Dolar Selandia Baru","td":"AUD/NZD","contract":100000,"tick":0.00001},
    "CADJPY":{"name":"Dolar Kanada / Yen Jepang","td":"CAD/JPY","contract":100000,"tick":0.001},
    "CHFJPY":{"name":"Franc Swiss / Yen Jepang","td":"CHF/JPY","contract":100000,"tick":0.001},
    "EURAUD":{"name":"Euro / Dolar Australia","td":"EUR/AUD","contract":100000,"tick":0.00001},
    "XAUUSD":{"name":"Gold / Dolar AS","td":"XAU/USD","contract":100,"tick":0.01},
    "XAGUSD":{"name":"Silver / Dolar AS","td":"XAG/USD","contract":5000,"tick":0.001},
    "AAPL":{"name":"Apple Inc.","td":"AAPL","contract":1,"tick":0.01},
    "MSFT":{"name":"Microsoft","td":"MSFT","contract":1,"tick":0.01},
    "NVDA":{"name":"NVIDIA","td":"NVDA","contract":1,"tick":0.01},
    "TSLA":{"name":"Tesla","td":"TSLA","contract":1,"tick":0.01},
    "AMZN":{"name":"Amazon","td":"AMZN","contract":1,"tick":0.01},
    "META":{"name":"Meta Platforms","td":"META","contract":1,"tick":0.01},
    "GOOG":{"name":"Alphabet / Google","td":"GOOG","contract":1,"tick":0.01},
}

INTERVALS = {
    "1 menit":"1min",
    "5 menit":"5min",
    "15 menit":"15min",
    "30 menit":"30min",
    "1 jam":"1h",
    "1 hari":"1day",
}

def load_secret(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)

def request_json(endpoint, params, timeout=12):
    r = requests.get(f"{TD_BASE}/{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def validate_key(api_key):
    try:
        j = request_json("price", {"symbol":"EUR/USD","apikey":api_key})
        if "price" in j:
            return True, f"API aktif. Test EUR/USD = {j['price']}"
        return False, str(j.get("message", j))
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=50, show_spinner=False)
def get_candles(symbol, interval, api_key, outputsize=500):
    try:
        j = request_json("time_series", {
            "symbol":symbol,
            "interval":interval,
            "outputsize":outputsize,
            "order":"ASC",
            "format":"JSON",
            "apikey":api_key,
        })
        vals = j.get("values", [])
        if not vals:
            return pd.DataFrame(), str(j.get("message", "No values"))
        df = pd.DataFrame(vals)
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["time"] = pd.to_datetime(df["datetime"])
        return df[["time","open","high","low","close","volume"]].dropna(subset=["open","high","low","close"]), ""
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=8, show_spinner=False)
def get_live_price(symbol, api_key):
    try:
        j = request_json("price", {"symbol":symbol,"apikey":api_key})
        if "price" not in j:
            return None, str(j.get("message", j))
        return float(j["price"]), ""
    except Exception as e:
        return None, str(e)

st.title("📈 Finex Live API Assistant V6")
st.caption("Market data terhubung ke Twelve Data API. BID/ASK Finex tetap Anda tentukan manual untuk sinkronisasi keputusan.")

with st.expander("⚙️ Koneksi API", expanded=True):
    env_key = load_secret("TWELVE_DATA_API_KEY","")
    api_key = st.text_input(
        "Twelve Data API Key",
        value=env_key,
        type="password",
        help="Untuk Streamlit Cloud, simpan sebagai TWELVE_DATA_API_KEY di App Secrets."
    )
    t1,t2 = st.columns([1,2])
    if t1.button("TEST API", use_container_width=True, disabled=not bool(api_key)):
        ok,msg = validate_key(api_key)
        st.success(msg) if ok else st.error(msg)
    t2.caption("API key hanya dipakai untuk mengambil market data dari Twelve Data. Password Finex tidak diperlukan.")

if not api_key:
    st.warning("Masukkan Twelve Data API Key terlebih dahulu.")
    st.stop()

a,b,c = st.columns([1.8,1,1])
with a:
    code = st.selectbox("Instrumen", list(CATALOG), format_func=lambda x:f"{x} — {CATALOG[x]['name']}")
with b:
    tf = st.selectbox("Timeframe", list(INTERVALS), index=2)
with c:
    mode = st.selectbox("Mode update", ["Hemat API (60 dtk)","Normal (20 dtk)","Cepat (10 dtk)"], index=0)

refresh_sec = {"Hemat API (60 dtk)":60,"Normal (20 dtk)":20,"Cepat (10 dtk)":10}[mode]
st_autorefresh(interval=refresh_sec*1000, key="refresh_v6")

spec = CATALOG[code]
td_symbol = spec["td"]
td_interval = INTERVALS[tf]

candles, candle_err = get_candles(td_symbol, td_interval, api_key)
live_price, price_err = get_live_price(td_symbol, api_key)

if candles.empty:
    st.error(f"Gagal mengambil candle {td_symbol}: {candle_err}")
    st.stop()

if live_price is None:
    live_price = float(candles.iloc[-1].close)
    st.warning(f"Endpoint live price gagal; memakai close candle terakhir. Detail: {price_err}")

st.markdown(
    f'<span class="badge">Twelve Data API</span>'
    f'<span class="badge">{td_symbol}</span>'
    f'<span class="badge">{tf}</span>'
    f'<span class="badge">Refresh {refresh_sec}s</span>',
    unsafe_allow_html=True
)

st.subheader("1. Harga live API")
m1,m2,m3 = st.columns(3)
m1.metric("Market price API", f"{live_price:,.5f}")
m2.metric("Candle terakhir", f"{float(candles.iloc[-1].close):,.5f}")
m3.metric("Update aplikasi", datetime.now().strftime("%H:%M:%S"))

st.subheader("2. BID / ASK Finex — manual")
st.caption("Masukkan BID dan ASK aktual yang Anda lihat sendiri di Finex Web MT5. Ini tetap menjadi acuan sinkronisasi ke broker.")
q1,q2,q3 = st.columns(3)
default_bid = float(st.session_state.get(f"{code}_bid", 0.0))
default_ask = float(st.session_state.get(f"{code}_ask", 0.0))
bid_in = q1.number_input("BID Finex", min_value=0.0, value=default_bid, format="%.5f")
ask_in = q2.number_input("ASK Finex", min_value=0.0, value=default_ask, format="%.5f")
if q3.button("SIMPAN BID/ASK", use_container_width=True):
    if bid_in > 0 and ask_in >= bid_in:
        st.session_state[f"{code}_bid"] = float(bid_in)
        st.session_state[f"{code}_ask"] = float(ask_in)
        st.session_state[f"{code}_sync_at"] = time.time()
        st.rerun()
    else:
        st.error("BID/ASK tidak valid.")

bid = float(st.session_state.get(f"{code}_bid",0.0))
ask = float(st.session_state.get(f"{code}_ask",0.0))
sync_at = float(st.session_state.get(f"{code}_sync_at",0.0))
sync_age = time.time()-sync_at if sync_at else 10**9
synced = bid>0 and ask>=bid and sync_age<=180

st.link_button("🔗 BUKA FINEX WEB MT5", FINEX_WEB_URL, use_container_width=True)

# Calibrate API market structure to Finex broker mid price.
ratio = 1.0
if synced:
    finex_mid = (bid+ask)/2
    ratio = finex_mid/live_price if live_price else 1.0

df = candles.copy()
for col in ["open","high","low","close"]:
    df[col] = df[col]*ratio

df = add_indicators(df)
sig = make_signal(df)
plan = make_plan(df, sig, atr_mult=1.5, rr=2.0)
regime = market_regime(df)
quality = signal_quality(df, sig, synced, sync_age)

if synced:
    display_market = ((bid+ask)/2)
else:
    display_market = live_price

st.subheader("3. Keputusan teknikal")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Harga acuan", f"{display_market:,.5f}")
k2.metric("Sinyal", sig["signal"])
k3.metric("Skor teknikal", f"{sig['score']}/5")
k4.metric("Setup quality", f"{quality['score']}/100")

if synced:
    st.success(f"✅ BID/ASK Finex tersinkron • usia {int(sync_age)} detik • spread {ask-bid:.5f}")
elif sync_at:
    st.warning("⚠️ BID/ASK Finex lebih dari 3 menit. Perbarui sebelum entry.")
else:
    st.warning("⚠️ Belum ada BID/ASK Finex. Analisis tetap berjalan dari API, tetapi entry belum dianggap siap.")

klass = {"BUY":"buy","SELL":"sell","WAIT":"wait"}[sig["signal"]]
st.markdown(f'<div class="hero {klass}">{sig["signal"]}</div>', unsafe_allow_html=True)

p1,p2,p3,p4 = st.columns(4)
fmt=lambda x:"-" if not np.isfinite(x) else f"{x:,.5f}"
p1.metric("Entry indikatif", fmt(plan["entry"]))
p2.metric("Stop Loss", fmt(plan["sl"]))
p3.metric("Take Profit", fmt(plan["tp"]))
p4.metric("Market regime", regime["label"])

st.subheader("Decision Gate")
checks = [
    ("BID/ASK Finex masih baru ≤3 menit", synced),
    ("Sinyal BUY/SELL", sig["signal"] != "WAIT"),
    ("Skor teknikal ≥4/5", sig["score"] >= 4),
    ("Setup quality ≥70/100", quality["score"] >= 70),
    ("Pasar tidak sideways", regime["tradeable"]),
]
for label, ok in checks:
    st.write(("✅" if ok else "❌"), label)

ready = all(ok for _,ok in checks)
if ready:
    st.success("SETUP SIAP DIPERTIMBANGKAN. Cek sekali lagi harga Finex dan risiko sebelum Anda memutuskan order.")
else:
    st.info("BELUM SIAP. Tidak perlu memaksakan entry.")

plot=df.tail(180)
fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.03,row_heights=[.62,.18,.20])
fig.add_trace(go.Candlestick(x=plot.time,open=plot.open,high=plot.high,low=plot.low,close=plot.close,name=code),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.ema_fast,name="EMA20"),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.ema_slow,name="EMA50"),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.rsi,name="RSI"),row=2,col=1)
fig.add_hline(y=70,row=2,col=1); fig.add_hline(y=30,row=2,col=1)
fig.add_trace(go.Bar(x=plot.time,y=plot.macd_hist,name="MACD Hist"),row=3,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.macd,name="MACD"),row=3,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.macd_signal,name="MACD Signal"),row=3,col=1)
fig.update_layout(height=660,xaxis_rangeslider_visible=False,legend_orientation="h",margin=dict(l=15,r=15,t=15,b=15))
st.plotly_chart(fig,use_container_width=True)

l,r = st.columns([1.1,1])
with l:
    st.subheader("Alasan sinyal")
    for x in sig["reasons"]:
        st.write("•",x)
with r:
    st.subheader("Kualitas setup")
    for x in quality["reasons"]:
        st.write("•",x)

st.subheader("4. Kalkulator risiko")
r1,r2,r3,r4 = st.columns(4)
balance = r1.number_input("Saldo akun (USD)", min_value=10.0, value=1000.0, step=100.0)
risk_pct = r2.slider("Risiko/trade (%)", 0.1, 2.0, 0.5, 0.1)
risk_money = balance*risk_pct/100
r3.metric("Risk money", f"${risk_money:,.2f}")
r4.metric("Contract size", f"{spec['contract']:,.0f}")

if np.isfinite(plan["sl"]):
    distance=abs(plan["entry"]-plan["sl"])
    if distance>0:
        lot=risk_money/(distance*spec["contract"])
        st.metric("Estimasi lot", f"{lot:.3f}")
        st.caption("Estimasi lot harus dicek lagi terhadap minimum volume, tick value, margin, dan contract specification di Finex.")

st.link_button("🔗 BUKA FINEX WEB MT5 UNTUK ORDER", FINEX_WEB_URL, use_container_width=True)

st.warning(
    "Aplikasi menggunakan API market data pihak ketiga, bukan feed broker Finex. "
    "BID/ASK Finex tetap Anda masukkan manual dan harus diperbarui sebelum keputusan. "
    "Setup quality bukan probabilitas profit dan tidak menjamin keuntungan."
)

with st.expander("Pemakaian API & mode hemat"):
    st.write(f"""
Refresh aktif: **{refresh_sec} detik**.

- **Hemat API (60 detik):** disarankan untuk akun API gratis.
- **Normal (20 detik):** lebih responsif, tetapi lebih banyak request.
- **Cepat (10 detik):** gunakan jika kuota API Anda memadai.

Candle disimpan cache sekitar 50 detik, sedangkan endpoint harga dapat diperbarui lebih cepat.
""")

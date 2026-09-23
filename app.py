
import os
import re
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
    page_title="Finex Auto Bid/Ask Assistant V7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TD_BASE = "https://api.twelvedata.com"
FINEX_INSTRUMENTS_URL = "https://finex.co.id/trading/instruments"
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

# Base catalog. Spread is overwritten from Finex site where available.
CATALOG = {
    "AUDCAD":{"name":"Dolar Australia / Dolar Kanada","td":"AUD/CAD","contract":100000,"tick":0.00001,"spread":0.00014},
    "AUDCHF":{"name":"Dolar Australia / Franc Swiss","td":"AUD/CHF","contract":100000,"tick":0.00001,"spread":0.00007},
    "AUDJPY":{"name":"Dolar Australia / Yen Jepang","td":"AUD/JPY","contract":100000,"tick":0.001,"spread":0.011},
    "AUDNZD":{"name":"Dolar Australia / Dolar Selandia Baru","td":"AUD/NZD","contract":100000,"tick":0.00001,"spread":0.00012},
    "AUDUSD":{"name":"Dolar Australia / Dolar AS","td":"AUD/USD","contract":100000,"tick":0.00001,"spread":0.00007},
    "CADJPY":{"name":"Dolar Kanada / Yen Jepang","td":"CAD/JPY","contract":100000,"tick":0.001,"spread":0.013},
    "CHFJPY":{"name":"Franc Swiss / Yen Jepang","td":"CHF/JPY","contract":100000,"tick":0.001,"spread":0.015},
    "EURAUD":{"name":"Euro / Dolar Australia","td":"EUR/AUD","contract":100000,"tick":0.00001,"spread":0.00011},
    "EURUSD":{"name":"Euro / Dolar AS","td":"EUR/USD","contract":100000,"tick":0.00001,"spread":0.00010},
    "GBPUSD":{"name":"Poundsterling / Dolar AS","td":"GBP/USD","contract":100000,"tick":0.00001,"spread":0.00012},
    "USDJPY":{"name":"Dolar AS / Yen Jepang","td":"USD/JPY","contract":100000,"tick":0.001,"spread":0.012},
    "USDCHF":{"name":"Dolar AS / Franc Swiss","td":"USD/CHF","contract":100000,"tick":0.00001,"spread":0.00012},
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
    "1 menit":"1min",
    "5 menit":"5min",
    "15 menit":"15min",
    "30 menit":"30min",
    "1 jam":"1h",
    "1 hari":"1day",
}

def secret(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)

API_KEY = secret("TWELVE_DATA_API_KEY","")

def request_json(endpoint, params, timeout=12):
    r=requests.get(f"{TD_BASE}/{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=8, show_spinner=False)
def latest_price(symbol, api_key):
    try:
        j=request_json("price",{"symbol":symbol,"apikey":api_key})
        if "price" not in j:
            return None, str(j.get("message",j))
        return float(j["price"]), ""
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=45, show_spinner=False)
def candles(symbol, interval, api_key, outputsize=500):
    try:
        j=request_json("time_series",{
            "symbol":symbol,"interval":interval,"outputsize":outputsize,
            "order":"ASC","format":"JSON","apikey":api_key
        })
        vals=j.get("values",[])
        if not vals:
            return pd.DataFrame(), str(j.get("message","No values"))
        df=pd.DataFrame(vals)
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns: df[c]=0
            df[c]=pd.to_numeric(df[c],errors="coerce")
        df["time"]=pd.to_datetime(df["datetime"])
        return df[["time","open","high","low","close","volume"]].dropna(subset=["open","high","low","close"]), ""
    except Exception as e:
        return pd.DataFrame(), str(e)

def parse_number(s):
    s=str(s).strip().replace("$","").replace(" ","")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+",s):
        return float(s.replace(".",""))
    if "," in s and "." not in s:
        s=s.replace(",",".")
    elif "," in s and "." in s:
        s=s.replace(".","").replace(",",".")
    return float(re.sub(r"[^0-9.\-]","",s) or 0)

@st.cache_data(ttl=21600, show_spinner=False)
def finex_spreads():
    out={}
    try:
        tables=pd.read_html(FINEX_INSTRUMENTS_URL)
        for t in tables:
            cols=[str(c).strip().lower() for c in t.columns]
            if not any("spread" in c for c in cols):
                continue
            # Locate expected columns by name.
            spread_idx=next((i for i,c in enumerate(cols) if "spread" in c),None)
            tick_idx=next((i for i,c in enumerate(cols) if "tick" in c),None)
            contract_idx=next((i for i,c in enumerate(cols) if "kontrak" in c),None)
            if spread_idx is None:
                continue
            for _,row in t.iterrows():
                first=str(row.iloc[0]).strip()
                m=re.match(r"([A-Z0-9.]+)",first)
                if not m: continue
                sym=m.group(1)
                try:
                    spread=parse_number(row.iloc[spread_idx])
                except Exception:
                    continue
                d={"spread":spread}
                if tick_idx is not None:
                    try:d["tick"]=parse_number(row.iloc[tick_idx])
                    except:pass
                if contract_idx is not None:
                    try:d["contract"]=parse_number(row.iloc[contract_idx])
                    except:pass
                out[sym]=d
        return out, "Finex web"
    except Exception:
        return {}, "fallback lokal"

st.title("📈 Finex Auto Bid/Ask Assistant V7")
st.caption(
    "BID/ASK dipilih otomatis dari market price API + spread Finex. "
    "Tidak perlu input BID/ASK manual."
)

with st.expander("⚙️ Koneksi API", expanded=not bool(API_KEY)):
    api_key=st.text_input("Twelve Data API Key", value=API_KEY, type="password")
    st.caption("Simpan API key di Streamlit Secrets sebagai TWELVE_DATA_API_KEY agar tidak perlu mengetik ulang.")

if not api_key:
    st.warning("Masukkan Twelve Data API Key.")
    st.stop()

a,b,c=st.columns([1.8,1,1])
with a:
    code=st.selectbox("Instrumen",list(CATALOG),format_func=lambda x:f"{x} — {CATALOG[x]['name']}")
with b:
    tf=st.selectbox("Timeframe",list(INTERVALS),index=2)
with c:
    refresh_mode=st.selectbox("Refresh",["60 detik","20 detik","10 detik"],index=0)

refresh_sec=int(refresh_mode.split()[0])
st_autorefresh(interval=refresh_sec*1000,key="refresh")

spec=CATALOG[code].copy()
spread_map,spread_source=finex_spreads()
if code in spread_map:
    spec.update(spread_map[code])

live,err=latest_price(spec["td"],api_key)
df,err2=candles(spec["td"],INTERVALS[tf],api_key)
if live is None:
    st.error(f"Gagal mengambil market price API: {err}")
    st.stop()
if df.empty:
    st.error(f"Gagal mengambil candle API: {err2}")
    st.stop()

# Auto-estimated bid/ask:
# midpoint = current live market price; spread = Finex published "spread umum".
spread=float(spec.get("spread") or max(float(spec["tick"])*10, abs(live)*0.00005))
auto_bid=live-spread/2
auto_ask=live+spread/2

st.markdown(
    f'<span class="badge">Live API: Twelve Data</span>'
    f'<span class="badge">Spread: {spread_source}</span>'
    f'<span class="badge">Auto BID/ASK</span>',
    unsafe_allow_html=True
)

st.subheader("1. Harga otomatis")
m1,m2,m3,m4=st.columns(4)
m1.metric("Market mid",f"{live:,.5f}")
m2.metric("Auto BID",f"{auto_bid:,.5f}")
m3.metric("Auto ASK",f"{auto_ask:,.5f}")
m4.metric("Spread acuan",f"{spread:,.5f}")

st.caption(
    "Auto BID/ASK adalah estimasi: market mid live/near-live dari Twelve Data "
    "+ spread umum Finex. Ini bukan quote executable langsung dari server broker."
)

# Align historical candles to latest live price.
last_close=float(df.iloc[-1].close)
ratio=live/last_close if last_close else 1.0
for col in ["open","high","low","close"]:
    df[col]=df[col]*ratio

df=add_indicators(df)
sig=make_signal(df)
plan=make_plan(df,sig,atr_mult=1.5,rr=2.0)
regime=market_regime(df)
quality=signal_quality(df,sig,True,0)

st.subheader("2. Sinyal & keputusan")
d1,d2,d3,d4=st.columns(4)
d1.metric("Sinyal",sig["signal"])
d2.metric("Skor teknikal",f"{sig['score']}/5")
d3.metric("Setup quality",f"{quality['score']}/100")
d4.metric("Regime",regime["label"])

klass={"BUY":"buy","SELL":"sell","WAIT":"wait"}[sig["signal"]]
st.markdown(f'<div class="hero {klass}">{sig["signal"]}</div>',unsafe_allow_html=True)

p1,p2,p3=st.columns(3)
fmt=lambda x:"-" if not np.isfinite(x) else f"{x:,.5f}"
p1.metric("Entry indikatif",fmt(plan["entry"]))
p2.metric("Stop Loss",fmt(plan["sl"]))
p3.metric("Take Profit",fmt(plan["tp"]))

st.subheader("Decision Gate")
checks=[
    ("Market price API tersedia", live is not None),
    ("Spread Finex tersedia/terestimasi", spread>0),
    ("Sinyal BUY/SELL", sig["signal"]!="WAIT"),
    ("Skor teknikal ≥4/5", sig["score"]>=4),
    ("Setup quality ≥70/100", quality["score"]>=70),
    ("Pasar tidak sideways", regime["tradeable"]),
]
for label,ok in checks:
    st.write(("✅" if ok else "❌"),label)

if all(ok for _,ok in checks):
    st.success("Setup memenuhi filter aplikasi. Tetap cek quote aktual di Finex sebelum order.")
else:
    st.info("Belum memenuhi seluruh filter. Jangan memaksakan entry.")

plot=df.tail(180)
fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.03,row_heights=[.62,.18,.20])
fig.add_trace(go.Candlestick(x=plot.time,open=plot.open,high=plot.high,low=plot.low,close=plot.close,name=code),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.ema_fast,name="EMA20"),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.ema_slow,name="EMA50"),row=1,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.rsi,name="RSI"),row=2,col=1)
fig.add_hline(y=70,row=2,col=1);fig.add_hline(y=30,row=2,col=1)
fig.add_trace(go.Bar(x=plot.time,y=plot.macd_hist,name="MACD Hist"),row=3,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.macd,name="MACD"),row=3,col=1)
fig.add_trace(go.Scatter(x=plot.time,y=plot.macd_signal,name="MACD Signal"),row=3,col=1)
fig.update_layout(height=660,xaxis_rangeslider_visible=False,legend_orientation="h",margin=dict(l=15,r=15,t=15,b=15))
st.plotly_chart(fig,use_container_width=True)

left,right=st.columns([1.1,1])
with left:
    st.subheader("Alasan sinyal")
    for x in sig["reasons"]: st.write("•",x)
with right:
    st.subheader("Kualitas setup")
    for x in quality["reasons"]: st.write("•",x)

st.subheader("3. Kalkulator risiko")
r1,r2,r3,r4=st.columns(4)
balance=r1.number_input("Saldo akun (USD)",min_value=10.0,value=1000.0,step=100.0)
risk_pct=r2.slider("Risiko/trade (%)",0.1,2.0,0.5,0.1)
risk_money=balance*risk_pct/100
r3.metric("Risk money",f"${risk_money:,.2f}")
r4.metric("Contract size",f"{spec['contract']:,.0f}")

if np.isfinite(plan["sl"]):
    distance=abs(plan["entry"]-plan["sl"])
    if distance>0:
        lot=risk_money/(distance*spec["contract"])
        st.metric("Estimasi lot",f"{lot:.3f}")
        st.caption("Verifikasi lot minimum, tick value, margin dan contract specification di Finex sebelum transaksi.")

st.link_button("🔗 BUKA FINEX WEB MT5",FINEX_WEB_URL,use_container_width=True)

with st.expander("Mode BID/ASK"):
    st.write("""
**AUTO (default):**  
BID = market mid API − ½ spread umum Finex  
ASK = market mid API + ½ spread umum Finex

Aplikasi memperbarui market mid secara otomatis. Spread umum Finex di-refresh dari halaman instrumen Finex secara berkala bila dapat dibaca.

**Penting:** ini adalah estimasi BID/ASK, bukan quote broker executable. Spread Finex bersifat floating, sehingga nilai aktual di aplikasi Finex/MT5 dapat berbeda, terutama saat volatilitas tinggi.
""")

st.warning(
    "Aplikasi membantu analisis, bukan menjamin profit. Auto BID/ASK menggunakan market data pihak ketiga "
    "dan spread umum Finex; quote aktual broker dapat berbeda. Untuk order real, cek harga akhir di Finex."
)

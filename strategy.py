
import numpy as np
import pandas as pd

def ema(s,span): return s.ewm(span=span,adjust=False).mean()
def rsi(s,p=14):
    d=s.diff();up=d.clip(lower=0);dn=(-d).clip(lower=0)
    au=up.ewm(alpha=1/p,adjust=False).mean();ad=dn.ewm(alpha=1/p,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)
def atr(df,p=14):
    prev=df.close.shift(1)
    tr=pd.concat([df.high-df.low,(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,adjust=False).mean()
def trend_strength(df,p=14):
    move=(df.close-df.close.shift(p)).abs()
    base=atr(df,p)*p
    return (100*move/base.replace(0,np.nan)).clip(0,100).fillna(0)
def add_indicators(df):
    x=df.copy()
    x["ema_fast"]=ema(x.close,20);x["ema_slow"]=ema(x.close,50)
    x["rsi"]=rsi(x.close);x["atr"]=atr(x)
    x["macd"]=ema(x.close,12)-ema(x.close,26)
    x["macd_signal"]=ema(x.macd,9);x["macd_hist"]=x.macd-x.macd_signal
    x["trend_strength"]=trend_strength(x)
    x["atr_pct"]=100*x.atr/x.close.replace(0,np.nan)
    return x
def make_signal(df):
    if len(df)<60:return {"signal":"WAIT","score":0,"reasons":["Data belum cukup"]}
    a,b=df.iloc[-1],df.iloc[-2];lp=sp=0;rs=[]
    if a.ema_fast>a.ema_slow:lp+=1;rs.append("EMA20 di atas EMA50")
    else:sp+=1;rs.append("EMA20 di bawah EMA50")
    if a.close>a.ema_fast:lp+=1;rs.append("Harga di atas EMA20")
    else:sp+=1;rs.append("Harga di bawah EMA20")
    if 50<a.rsi<70:lp+=1;rs.append(f"RSI {a.rsi:.1f}: momentum naik")
    elif 30<a.rsi<50:sp+=1;rs.append(f"RSI {a.rsi:.1f}: momentum turun")
    else:rs.append(f"RSI {a.rsi:.1f}: ekstrem/netral")
    if a.macd_hist>0 and a.macd_hist>=b.macd_hist:lp+=1;rs.append("MACD positif/menguat")
    elif a.macd_hist<0 and a.macd_hist<=b.macd_hist:sp+=1;rs.append("MACD negatif/melemah")
    else:rs.append("MACD belum kuat")
    if a.close>a.open:lp+=1;rs.append("Candle terakhir bullish")
    elif a.close<a.open:sp+=1;rs.append("Candle terakhir bearish")
    if lp>=4 and lp>=sp+2:return {"signal":"BUY","score":lp,"reasons":rs}
    if sp>=4 and sp>=lp+2:return {"signal":"SELL","score":sp,"reasons":rs}
    return {"signal":"WAIT","score":max(lp,sp),"reasons":rs}
def make_plan(df,sig,atr_mult=1.5,rr=2.0):
    a=df.iloc[-1];entry=float(a.close);av=float(a.atr)
    if sig["signal"]=="BUY":sl=entry-atr_mult*av;tp=entry+rr*(entry-sl)
    elif sig["signal"]=="SELL":sl=entry+atr_mult*av;tp=entry-rr*(sl-entry)
    else:sl=np.nan;tp=np.nan
    return {"entry":entry,"sl":sl,"tp":tp}
def market_regime(df):
    a=df.iloc[-1];t=float(a.trend_strength);v=float(a.atr_pct) if np.isfinite(a.atr_pct) else 0
    if t>=35:label="Trending";tradeable=True
    elif t>=20:label="Mixed";tradeable=True
    else:label="Sideways";tradeable=False
    if v>3:label+=" / High vol"
    elif v<0.1:label+=" / Low vol"
    return {"label":label,"tradeable":tradeable}
def signal_quality(df,sig,synced=True,sync_age=0):
    a=df.iloc[-1];score=25 if synced else 0;rs=["Live market data aktif (+25)" if synced else "Live market data tidak aktif (+0)"]
    pts=min(sig["score"],5)*10;score+=pts;rs.append(f"Teknikal {sig['score']}/5 (+{pts})")
    ts=float(a.trend_strength)
    if ts>=35:score+=15;rs.append("Trend strength kuat (+15)")
    elif ts>=20:score+=8;rs.append("Trend strength sedang (+8)")
    else:rs.append("Trend strength lemah (+0)")
    ap=float(a.atr_pct) if np.isfinite(a.atr_pct) else 0
    if 0.1<=ap<=3:score+=10;rs.append("Volatilitas layak (+10)")
    elif ap>3:score+=3;rs.append("Volatilitas tinggi (+3)")
    else:rs.append("Volatilitas rendah (+0)")
    if sig["signal"]=="WAIT":score=min(score,49)
    return {"score":int(min(score,100)),"reasons":rs}

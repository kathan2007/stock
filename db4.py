# # ------------------------------------------------------------
# # AI Stock Dashboard – plots Spot vs Prediction (5 m & 1 h)
# # v2025‑06‑29d
# # ------------------------------------------------------------
# import os, io, warnings, sqlite3, functools, requests, joblib, yfinance as yf
# import ta, pandas as pd, numpy as np
# from datetime import datetime, timedelta, date
# from pytz import timezone
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import MinMaxScaler
# import streamlit as st

# try:
#     from streamlit_autorefresh import st_autorefresh
# except ModuleNotFoundError:
#     st_autorefresh = lambda *a, **k: None

# warnings.filterwarnings("ignore")
# st.set_page_config(page_title="AI Stock Dashboard — Spot vs Prediction", layout="wide")

# # ---------------- CONSTANTS -----------------
# LOOKBACK_ST, LOOKBACK_LT   = 30, 30   # bars
# HORIZON_LT                 = 20       # 20×1 h ≈ 1 trading day ahead
# MAX_ROWS_ST, MAX_ROWS_LT   = 1000, 800
# EST_ST, EST_LT, DEPTH      = 80, 120, 2
# DB_PATH, CACHE_DIR         = "stock_trades.db", "model_cache"
# REFRESH_SEC                = 60
# AUTO_LOG_EVERY             = 180
# IST                        = timezone("Asia/Kolkata")
# os.makedirs(CACHE_DIR, exist_ok=True)

# FEATURES = ["Close","RSI","MACD","EMA","Super"]
# def safe_rerun(): (st.rerun if hasattr(st,"rerun") else st.experimental_rerun)()

# # ---------------- SYMBOL UNIVERSE ----------
# @functools.lru_cache(maxsize=1)
# def fetch_nse_symbols():
#     url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
#     try:
#         df = pd.read_csv(io.BytesIO(requests.get(url, timeout=6).content))
#         return sorted(df["SYMBOL"].str.strip().str.upper() + ".NS")
#     except Exception:
#         return sorted(["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","^NSEI"])
# ALL_SYMBOLS = fetch_nse_symbols()

# # ---------------- DB -----------------------
# def init_db():
#     with sqlite3.connect(DB_PATH) as c:
#         c.execute("""CREATE TABLE IF NOT EXISTS pnl(
#             date TEXT, stock TEXT, spot REAL,
#             pred_short REAL, pred_long REAL,
#             action TEXT, entry REAL, exit_price REAL, pnl REAL)""")

# def log_trade(sym, spot, ps, pl, act):
#     now = datetime.now(IST).isoformat(sep=" ", timespec="seconds")
#     with sqlite3.connect(DB_PATH) as c:
#         if act=="BUY":
#             c.execute("""INSERT INTO pnl(date,stock,spot,pred_short,pred_long,action,entry)
#                          VALUES (?,?,?,?,?,?,?)""",(now,sym,spot,ps,pl,act,spot))
#         elif act=="SELL":
#             rid_ent=c.execute("""SELECT rowid,entry FROM pnl
#                                  WHERE stock=? AND action='BUY' AND exit_price IS NULL
#                                  ORDER BY date DESC LIMIT 1""",(sym,)).fetchone()
#             if rid_ent:
#                 rid, ent = rid_ent
#                 pnl = round(spot-ent,2)
#                 c.execute("UPDATE pnl SET exit_price=?, pnl=? WHERE rowid=?",(spot,pnl,rid))
#             c.execute("""INSERT INTO pnl(date,stock,spot,pred_short,pred_long,action)
#                          VALUES (?,?,?,?,?,?)""",(now,sym,spot,ps,pl,act))
#         else:
#             c.execute("""INSERT INTO pnl(date,stock,spot,pred_short,pred_long,action)
#                          VALUES (?,?,?,?,?,?)""",(now,sym,spot,ps,pl,act))

# def clear_logs():
#     with sqlite3.connect(DB_PATH) as c:
#         c.execute("DELETE FROM pnl")

# # ------------- DATA UTILS ------------------
# def flatten(df):
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(-1)
#     df.columns = [str(c).strip() for c in df.columns]
#     return df

# def extract_close(df):
#     df = flatten(df)
#     lc = {c.lower():c for c in df.columns}
#     for k in ("close","adj close"):
#         if k in lc: return pd.to_numeric(df[lc[k]], errors="coerce")
#     return pd.to_numeric(df.select_dtypes("number").iloc[:,0], errors="coerce")

# def add_indicators(df):
#     df = flatten(df)
#     c = extract_close(df)
#     df["Close"]=c
#     df["RSI"]=ta.momentum.RSIIndicator(c).rsi()
#     df["MACD"]=ta.trend.MACD(c).macd_diff()
#     df["EMA"]=ta.trend.EMAIndicator(c,20).ema_indicator()
#     df["Super"]=ta.trend.STCIndicator(c).stc()
#     return df.dropna()

# def prep_arrays(df, lookback, horizon, max_rows):
#     df = df.iloc[-max_rows:] if len(df)>max_rows else df
#     sc  = MinMaxScaler()
#     arr = sc.fit_transform(df[FEATURES])
#     X,y=[],[]
#     for i in range(lookback,len(arr)-horizon):
#         X.append(arr[i-lookback:i].ravel())
#         y.append(arr[i+horizon][0])
#     return np.array(X), np.array(y), sc, df

# def load_or_train(path, fn):
#     today=date.today().isoformat(); meta=path+".meta"
#     if os.path.exists(path) and os.path.exists(meta) and open(meta).read()==today:
#         return joblib.load(path)
#     obj=fn(); joblib.dump(obj,path); open(meta,"w").write(today); return obj

# # ------------- MODEL LOADERS ---------------
# def get_st_model(sym):
#     p=os.path.join(CACHE_DIR,f"st_{sym}.joblib".replace(".","_"))
#     def train():
#         raw=yf.download(sym,interval="5m",period="7d",progress=False)
#         if raw.empty: raw=yf.download(sym,interval="1d",period="30d")
#         X,y,sc,df=prep_arrays(add_indicators(raw),LOOKBACK_ST,0,MAX_ROWS_ST)
#         m=GradientBoostingRegressor(n_estimators=EST_ST,max_depth=DEPTH).fit(X,y)
#         return m,sc,df
#     return load_or_train(p,train)

# def get_lt_model(sym):
#     p=os.path.join(CACHE_DIR,f"lt_{sym}.joblib".replace(".","_"))
#     def train():
#         raw=yf.download(sym,interval="60m",period="90d",progress=False)
#         if raw.empty: raw=yf.download(sym,interval="1h",period="90d")
#         X,y,sc,df=prep_arrays(add_indicators(raw),LOOKBACK_LT,HORIZON_LT,MAX_ROWS_LT)
#         m=GradientBoostingRegressor(n_estimators=EST_LT,max_depth=DEPTH).fit(X,y)
#         return m,sc,df
#     return load_or_train(p,train)

# def predict(m,sc,df,look):
#     if m is None or len(df)<look: return None
#     vec=sc.transform(df[FEATURES].iloc[-look:]).ravel().reshape(1,-1)
#     p_scaled=m.predict(vec)[0]
#     return sc.inverse_transform([[p_scaled,0,0,0,0]])[0][0]

# def get_action(spot,pred):
#     if pred is None: return "HOLD"
#     d=pred-spot
#     return "BUY" if d>0.5 else "SELL" if d<-0.5 else "HOLD"

# # ------------- DASHBOARD --------------------
# def dashboard():
#     init_db()
#     st.header("🇮🇳 AI Stock Dashboard – Spot vs Prediction")

#     sym=st.selectbox("Stock",["ALL"]+ALL_SYMBOLS,
#                      index=ALL_SYMBOLS.index("RELIANCE.NS") if "RELIANCE.NS" in ALL_SYMBOLS else 0)

#     if sym!="ALL":
#         m_st, sc_st, df_st = get_st_model(sym)
#         m_lt, sc_lt, df_lt = get_lt_model(sym)

#         spot = float(df_st["Close"].iloc[-1])
#         pred_short = predict(m_st, sc_st, df_st, LOOKBACK_ST)
#         pred_long  = predict(m_lt, sc_lt, df_lt, LOOKBACK_LT)
#         action = get_action(spot, pred_short)

#         # ------ Metrics
#         c1,c2,c3,c4 = st.columns(4)
#         c1.metric("Spot",f"₹{spot:.2f}")
#         c2.metric("Short‑term (5 m)",f"{pred_short:.2f}" if pred_short else "N/A",
#                   f"{pred_short-spot:+.2f}" if pred_short else "")
#         c3.metric("Long‑term (1 h)", f"{pred_long:.2f}" if pred_long else "N/A",
#                   f"{pred_long-spot:+.2f}" if pred_long else "")
#         c4.metric("Action",action)

#         # ------ Plot 1: 5‑minute
#         df_plot = df_st[["Close"]].tail(120).copy()
#         df_plot.rename(columns={"Close":"Spot"}, inplace=True)
#         df_plot["Predicted"]=np.nan
#         df_plot.loc[df_plot.index[-1],"Predicted"]=pred_short
#         st.subheader("5‑minute Spot vs Prediction")
#         st.line_chart(df_plot)

#         # ------ Plot 2: 1‑hour
#         df_plot2 = df_lt[["Close"]].tail(40).copy()
#         df_plot2.rename(columns={"Close":"Spot"}, inplace=True)
#         df_plot2["Predicted"]=np.nan
#         df_plot2.loc[df_plot2.index[-1],"Predicted"]=pred_long
#         st.subheader("1‑hour Spot vs Prediction")
#         st.line_chart(df_plot2)

#         # ------ Auto‑log (throttled)
#         now_aware = datetime.now(IST)
#         with sqlite3.connect(DB_PATH) as c:
#             last=c.execute("SELECT date FROM pnl WHERE stock=? ORDER BY date DESC LIMIT 1",
#                            (sym,)).fetchone()
#         if not last or datetime.fromisoformat(last[0]) < now_aware-timedelta(seconds=AUTO_LOG_EVERY):
#             log_trade(sym, spot, pred_short, pred_long, action)

#     # ------ Logs table
#     st.subheader("Logs")
#     q="SELECT * FROM pnl"
#     if sym!="ALL": q+=f" WHERE stock='{sym}'"
#     q+=" ORDER BY date DESC LIMIT 400"
#     logs=pd.read_sql(q,sqlite3.connect(DB_PATH))
#     order=["date","stock","action","spot","pred_short","pred_long","entry","exit_price","pnl"]
#     logs=logs[[c for c in order if c in logs.columns]+[c for c in logs.columns if c not in order]]
#     st.dataframe(logs,use_container_width=True)

#     # buttons
#     colA,colB=st.columns(2)
#     if colA.button("🧹 Clear Logs"):
#         clear_logs(); safe_rerun()
#     if colB.button("♻️ Clear Cache"):
#         for f in os.listdir(CACHE_DIR): os.remove(os.path.join(CACHE_DIR,f))
#         safe_rerun()

# # ------------- MAIN -------------------------
# if __name__=="__main__":
#     st_autorefresh(interval=REFRESH_SEC*1000,key="refresh")
#     dashboard()


# --------------------------------------------------------------------
# Indian Stocks Dashboard – 5 m • 1 h • 1 d Forecasts
# Safe intraday periods (1 m max 7 d, 5 m max 60 d)
# --------------------------------------------------------------------
import streamlit as st, pandas as pd, yfinance as yf, plotly.graph_objects as go
import re, sqlite3, numpy as np, warnings
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from streamlit_autorefresh import st_autorefresh
import ta
warnings.filterwarnings("ignore")

try:
    import pandas_ta as pta
    HAS_PTA = True
except ImportError:
    HAS_PTA = False

DB = "stock_log.db"

# ---------- DB helpers ------------------------------------------------
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS log(
            ts TEXT, ticker TEXT, horizon TEXT,
            current REAL, predicted REAL, signal TEXT)""")
        c.commit()

def log_pred(t,h,cur,pred,sig):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO log VALUES (?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), t, h, cur, pred, sig))
        c.commit()

def fetch_log():
    with sqlite3.connect(DB) as c:
        return pd.read_sql("SELECT * FROM log ORDER BY ts DESC", c)

def clear_log():
    with sqlite3.connect(DB) as c:
        c.execute("DELETE FROM log"); c.commit()

# ---------- cap helper ------------------------------------------------
MAX_DAYS = {
    "1m": 7, "2m": 7,
    "5m": 60, "15m": 60, "30m": 60, "60m": 60,
}

def cap_period(requested: str, interval: str) -> str:
    """Return a period string that does not exceed Yahoo's limit."""
    if interval not in MAX_DAYS or not requested.endswith("d"):
        return requested
    req_days = int(requested[:-1])
    if req_days > MAX_DAYS[interval]:
        return f"{MAX_DAYS[interval]}d"
    return requested

# ---------- yfinance column normaliser -------------------------------
def normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if {"Open","High","Low","Close"}.issubset(df.columns.get_level_values(0)):
            df = df.droplevel(1, axis=1)
        else:
            df = df.xs(ticker, level=0, axis=1, drop_level=True)
    mapper = {"open":"Open","high":"High","low":"Low",
              "close":"Close","adjclose":"Adj Close","volume":"Volume"}
    rn={}
    for col in df.columns:
        core=re.split(r"[._\\-]", re.sub(r"\s+","",col).lower())[-1]
        if core in mapper: rn[col]=mapper[core]
    return df.rename(columns=rn).loc[:,~df.columns.duplicated()]

# ---------- resilient fetch ------------------------------------------
def yf_fetch(ticker: str, period: str, interval: str):
    """Download with period capped; retry with smaller periods if needed."""
    period = cap_period(period, interval)
    def _get(per):
        return yf.download(ticker, period=per, interval=interval,
                           progress=False, threads=False, group_by="ticker",
                           auto_adjust=False)
    data = _get(period)
    # if empty and we can still shrink (1m/2m only)
    if data.empty and interval in {"1m","2m"} and period != "7d":
        data = _get("7d")
    # final fallback: daily candles
    if data.empty and interval != "1d":
        data = yf.download(ticker, period="365d", interval="1d",
                           progress=False, threads=False, auto_adjust=False)
        interval = "1d"
    return data, interval

# ---------- feature engineering + forecaster -------------------------
def build_features(df):
    d=df.copy()
    d["ret"]=d["Close"].pct_change().fillna(0)
    d["rsi"]=ta.momentum.rsi(d["Close"],14).fillna(method="ffill").fillna(method="bfill")
    d["ema"]=d["Close"].ewm(span=20).mean().fillna(method="ffill")
    d["stoch"]=ta.momentum.StochasticOscillator(
        d["High"],d["Low"],d["Close"],14,3).stoch().fillna(method="ffill")
    return d

def forecast(df, lookback, fallback):
    df=build_features(df).dropna()
    if len(df)<30:
        return fallback
    look=min(lookback, len(df)-1)
    feats=["Close","ret","rsi","ema","stoch"]
    X=df[feats].iloc[-look-1:-1].values
    y=df["Close"].iloc[-look:].values
    try:
        gb=GradientBoostingRegressor(n_estimators=250,max_depth=3,
                                     learning_rate=0.05,random_state=1)
        gb.fit(X,y)
        return float(gb.predict(df[feats].iloc[[-1]].values)[0])
    except Exception:
        return fallback

# ---------- pattern helper -------------------------------------------
def patterns(df):
    if not HAS_PTA: return {}
    out={}
    for cd,lb in {"hammer":"Hammer","doji":"Doji","engulfing":"Engulfing"}.items():
        try:
            sig=pta.cdl_pattern(df["Open"],df["High"],df["Low"],df["Close"],name=cd)
            idx=sig[sig!=0].index.tolist()
            if idx: out[lb]=idx
        except: pass
    return out

# ---------- chart builder --------------------------------------------
def candle_chart(df, pats, title):
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=df["Datetime"],open=df["Open"],high=df["High"],
                                 low=df["Low"],close=df["Close"],name="Candles"))
    df["EMA20"]=df["Close"].ewm(20).mean()
    df["EMA50"]=df["Close"].ewm(50).mean()
    fig.add_trace(go.Scatter(x=df["Datetime"],y=df["EMA20"],line=dict(color="orange"),name="EMA20"))
    fig.add_trace(go.Scatter(x=df["Datetime"],y=df["EMA50"],line=dict(color="blue"),name="EMA50"))
    for p,rows in pats.items():
        for r in rows:
            fig.add_trace(go.Scatter(x=[df.at[r,"Datetime"]],y=[df.at[r,"Close"]],
                                     mode="markers+text",marker=dict(color="purple",size=9),
                                     text=[p],textposition="top center",showlegend=False))
    fig.update_layout(title=title,height=650,xaxis_rangeslider_visible=False)
    return fig

# ---------- NSE list --------------------------------------------------
@st.cache_data(show_spinner=False)
def nse_list():
    url="https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df=pd.read_csv(url,on_bad_lines="skip")[["SYMBOL","NAME OF COMPANY"]]
    df["Ticker"]=df["SYMBOL"].str.strip()+".NS"
    return df.sort_values("SYMBOL")

# ---------- Streamlit app --------------------------------------------
def main():
    st.set_page_config("Indian Stocks Dashboard", layout="wide")
    st_autorefresh(interval=60000, key="refresh")
    init_db()

    st.title("📈 Indian Stocks Dashboard – 5 m • 1 h • 1 d Forecasts")

    nse=nse_list()
    ticker=st.selectbox("Select stock", nse["Ticker"],
        format_func=lambda t:f"{t} – {nse.loc[nse['Ticker']==t,'NAME OF COMPANY'].values[0]}")

    tf=st.selectbox("Chart interval",["1m","2m","5m","15m","30m","60m","1h","1d"])
    # choose default period without violating cap
    default_period = {"1m":"7d", "2m":"7d",
                      "5m":"60d","15m":"60d","30m":"60d","60m":"60d",
                      "1h":"365d","1d":"365d"}.get(tf,"60d")

    raw, tf_eff = yf_fetch(ticker, default_period, tf)
    if raw.empty:
        st.error("Data unavailable (even after fallback). Try another symbol/interval.")
        return
    df = normalise(raw, ticker).reset_index(names="Datetime").dropna(subset=["Open","High","Low","Close"])
    df["Datetime"]=pd.to_datetime(df["Datetime"])

    # prediction datasets
    df5,_  = yf_fetch(ticker,"7d","5m")
    df1,_  = yf_fetch(ticker,"60d","1h")
    dfD,_  = yf_fetch(ticker,"365d","1d")
    df5=normalise(df5,ticker).dropna()
    df1=normalise(df1,ticker).dropna()
    dfD=normalise(dfD,ticker).dropna()

    live=yf.Ticker(ticker).info.get("regularMarketPrice") or df["Close"].iloc[-1]

    pred5 = forecast(df5,  80, live)
    pred1 = forecast(df1, 120, live)
    predD = forecast(dfD, 200, live)

    def sig(p):
        diff=(p-live)/live*100
        return ("BUY","green") if diff>=2 else ("SELL","red") if diff<=-2 else ("HOLD","orange")
    s5,c5=sig(pred5); s1,c1=sig(pred1); sD,cD=sig(predD)

    log_pred(ticker,"5m",live,pred5,s5)
    log_pred(ticker,"1h",live,pred1,s1)
    log_pred(ticker,"1d",live,predD,sD)

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Live Price",f"₹ {live:,.2f}")
    col2.metric("Next 5 m",f"₹ {pred5:,.2f}"); col2.markdown(f"<span style='color:{c5};font‑weight:bold'>{s5}</span>",True)
    col3.metric("Next 1 h",f"₹ {pred1:,.2f}"); col3.markdown(f"<span style='color:{c1};font‑weight:bold'>{s1}</span>",True)
    col4.metric("Next 1 d",f"₹ {predD:,.2f}"); col4.markdown(f"<span style='color:{cD};font‑weight:bold'>{sD}</span>",True)

    st.plotly_chart(candle_chart(df, patterns(df), f"{ticker} – {tf_eff} Candles"),
                    use_container_width=True)

    st.subheader("Prediction Log")
    st.dataframe(fetch_log(), height=350, use_container_width=True)
    if st.button("Clear Log"): clear_log(); st.success("Log cleared.")

    st.caption("1‑minute data limited to 7 days, 5‑minute to 60 days by Yahoo. "
               "App auto‑caps period and retries; falls back to daily candles if needed. "
               "Forecasts use GradientBoosting on technical features.")

if __name__=="__main__":
    main()
 
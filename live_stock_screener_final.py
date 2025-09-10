# In file: ~/Documents/live_stock_screener_final.py

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
import nltk
import numpy as np
import math
from newsapi import NewsApiClient
from kiteconnect import KiteConnect
import ssl
import plotly.graph_objects as go
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import os
import google.generativeai as genai

# --- Definitive, thread-safe method to silence yfinance 404 errors ---
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(page_title="Rohan's Market Strategy ", page_icon="📈", layout="wide")

# --- SECURE API KEY MANAGEMENT ---
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    KITE_API_KEY = st.secrets["KITE_API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except FileNotFoundError:
    st.error("⚠️ `secrets.toml` file not found. Please create it in a `.streamlit` folder.", icon="🚨")
    st.stop()
except KeyError as e:
    st.error(f"⚠️ The `{e}` key is missing from your `secrets.toml` file.", icon="🚨")
    st.stop()

# Configure the Generative AI library
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- One-time setup ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_data():
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        st.info("First-time setup: Downloading sentiment analysis model...")
        nltk.download('vader_lexicon')

download_nltk_data()

# --- Functions to Save and Load Portfolio from a File ---
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio_from_disk():
    if not os.path.exists(PORTFOLIO_FILE): return []
    try:
        with open(PORTFOLIO_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return []

def save_portfolio_to_disk(portfolio_data):
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(portfolio_data, f, indent=4)

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_disk()

# --- Header & Sidebar ---
st.title(" Rohan's Market Strategy ")
st.sidebar.title("⚙️ Strategy Parameters")
st.sidebar.subheader("Technical Indicators")
RSI_OVERBOUGHT = st.sidebar.slider("RSI Overbought Level", 50, 100, 70, 1)
ATR_MULTIPLIER = st.sidebar.number_input("ATR Multiplier for Stop-Loss", 1.0, 5.0, 2.0, 0.5)
VOLUME_MULTIPLIER = st.sidebar.number_input("Volume Spike Multiplier", 1.0, 5.0, 1.5, 0.1)

st.sidebar.subheader("Fundamental Filters") # NEW
MAX_PE_RATIO = st.sidebar.number_input("Maximum P/E Ratio", 1, 200, 50, 5)
MIN_ROE = st.sidebar.number_input("Minimum Return on Equity (ROE %)", -50, 100, 15, 1)

st.sidebar.subheader("General Filters")
MIN_PRICE = st.sidebar.slider("Minimum Stock Price (₹)", 10, 500, 50, 10)
MIN_VOLUME = st.sidebar.slider("Minimum 20-Day Avg Volume", 10000, 1000000, 100000, 10000)

# --- UTILITY FUNCTIONS ---
def get_watchlist(index_name):
    # (No changes to this function)
    watchlists = {
        "Bank Nifty": ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'AUBANK', 'BANDHANBNK'],
        "Nifty 50": ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR', 'BHARTIARTL', 'ITC', 'LTIM', 'SBIN', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'TATAMOTORS', 'ADANIENT', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'TATASTEEL', 'AXISBANK', 'ONGC', 'NTPC', 'BAJAJFINSV', 'ADANIPORTS', 'NESTLEIND', 'COALINDIA', 'WIPRO', 'POWERGRID', 'M&M', 'GRASIM', 'JSWSTEEL', 'HINDALCO', 'ULTRACEMCO', 'EICHERMOT', 'DRREDDY', 'CIPLA', 'INDUSINDBK', 'BRITANNIA', 'HEROMOTOCO', 'APOLLOHOSP', 'DIVISLAB', 'BPCL', 'SHREECEM', 'UPL', 'SBILIFE', 'TECHM', 'BAJAJ-AUTO', 'ADANIGREEN', 'TATACONSUM'],
        "Nifty IT": ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'LTIM', 'TECHM', 'PERSISTENT', 'OFSS', 'MPHASIS', 'COFORGE']
    }
    return watchlists.get(index_name, [])

def get_fallback_stocks():
    # (No changes to this function)
    return ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR', 'BHARTIARTL', 'ITC', 'LTIM', 'SBIN', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'TATAMOTORS', 'ADANIENT', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'TATASTEEL', 'AXISBANK', 'ONGC', 'NTPC', 'BAJAJFINSV', 'ADANIPORTS', 'NESTLEIND', 'COALINDIA', 'WIPRO', 'POWERGRID', 'M&M', 'GRASIM', 'JSWSTEEL', 'HINDALCO', 'ULTRACEMCO', 'EICHERMOT', 'DRREDDY', 'CIPLA', 'INDUSINDBK', 'BRITANNIA', 'HEROMOTOCO', 'APOLLOHOSP', 'DIVISLAB', 'BPCL', 'SHREECEM', 'UPL', 'SBILIFE', 'TECHM', 'BAJAJ-AUTO', 'ADANIGREEN', 'TATACONSUM', 'ZOMATO', 'PAYTM', 'POLICYBZR', 'NYKAA', 'VEDL', 'ITC', 'IOC', 'BHEL', 'GAIL', 'SAIL', 'DLF']

@st.cache_data(ttl=86400)
def _load_nse_stocks_cached(api_key):
    # (No changes to this function)
    try:
        kite = KiteConnect(api_key=api_key)
        instruments = kite.instruments("NSE")
        equity_symbols = [inst['tradingsymbol'] for inst in instruments if isinstance(inst, dict) and inst.get('instrument_type') == 'EQ']
        if not equity_symbols:
            return get_fallback_stocks()
        return sorted(equity_symbols)
    except Exception:
        st.sidebar.warning(f"Could not connect to Zerodha. Using a fallback list.")
        return get_fallback_stocks()

def create_stock_chart(df, stock_symbol):
    # (No changes to this function)
    fig = go.Figure()
    if 'Close' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
    if 'SMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange', dash='dash')))
    if 'SMA200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'{stock_symbol} Chart', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- HEAVILY MODIFIED FUNCTION ---
@st.cache_data(ttl=3600)
def analyze_stock(stock_symbol, bulk_data_frame, atr_multiplier, rsi_overbought, volume_multiplier):
    try:
        # --- Technical Analysis (Existing Logic) ---
        ticker_str = f"{stock_symbol.upper()}.NS"
        df = bulk_data_frame.xs(ticker_str, level=1, axis=1).copy() if isinstance(bulk_data_frame.columns, pd.MultiIndex) else bulk_data_frame.copy()
        df.dropna(how='all', inplace=True)
        if df.empty or len(df) < 200: return None
        
        has_volume = 'Volume' in df.columns and pd.to_numeric(df['Volume'], errors='coerce').notna().any() and pd.to_numeric(df['Volume'], errors='coerce').sum() > 0
        df.ta.sma(length=50, append=True, col_names=("SMA50",)); df.ta.sma(length=200, append=True, col_names=("SMA200",))
        df.ta.rsi(length=14, append=True, col_names=("RSI",)); df.ta.atr(length=14, append=True, col_names=("ATR",))
        if has_volume: df.ta.sma(close='Volume', length=20, append=True, col_names=("Avg Volume",))
        
        chart_df = df.copy()
        df.dropna(inplace=True)
        if df.empty: return None
        
        last = df.iloc[-1]
        price = last['Close']
        volume_ratio = (last['Volume'] / last['Avg Volume']) if has_volume and 'Avg Volume' in last and last['Avg Volume'] > 0 else 0
        has_spike = volume_ratio >= volume_multiplier
        strategy, reason = ("N/A", "-")
        if price > last['SMA200']:
            if last['RSI'] > rsi_overbought: strategy, reason = ("🟡 HOLD", "RSI Overbought")
            elif not has_spike: strategy, reason = ("🟡 HOLD", "Volume Too Low")
            elif price > last['SMA50']: strategy, reason = ("📈 Momentum Breakout", "Price > SMA50 & SMA200")
            else: strategy, reason = ("🟢 Buy the Dip", "Above SMA200, Below SMA50")
        elif price > last['SMA50']: strategy, reason = ("🟡 WATCH", "Nearing Uptrend (Price > SMA50)")
        else: strategy, reason = ("🔴 AVOID", "Price Below Key SMAs")
        
        # --- NEW: Fundamental Analysis ---
        pe_ratio, pb_ratio, roe, debt_to_equity = np.nan, np.nan, np.nan, np.nan
        try:
            info = yf.Ticker(ticker_str).info
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            debt_to_equity = info.get('debtToEquity', np.nan)
        except Exception:
            pass # Silently fail if fundamental data is not available

        df_30 = chart_df.tail(30)
        return {
            "Stock Name": stock_symbol, "Price": price, "Strategy": strategy, "RSI": last['RSI'],
            "Entry": df_30['Low'].min() if strategy == "🟢 Buy the Dip" else price,
            "Target": df_30['High'].max(), "Stop-Loss": price - (atr_multiplier * last['ATR']),
            "Suggested Hold (Days)": abs((df_30['High'].idxmax() - df_30['Low'].idxmin()).days) or 1,
            "Vol Ratio": volume_ratio, "Chart Data": chart_df, "Reason": reason,
            # NEW fundamental data points added
            "P/E": pe_ratio, "P/B": pb_ratio, "ROE %": roe * 100 if pd.notna(roe) else np.nan, "D/E": debt_to_equity
        }
    except (KeyError, IndexError, TypeError): 
        return None

def fetch_single_batch(batch, period):
    # (No changes to this function)
    tickers = [f"{s.upper()}.NS" for s in batch]
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True, threads=False)
    if data.empty: return None, batch
    failed = [t.replace('.NS', '') for t in tickers if t.upper() not in data.columns.get_level_values(1).unique()]
    return data, failed

def fetch_market_data(symbols, period="5y", progress_text="Downloading data..."):
    # (No changes to this function)
    if not symbols: return None
    all_data, failed_symbols = [], []
    batches = [symbols[i:i+100] for i in range(0, len(symbols), 100)]
    progress_bar = st.progress(0, text=f"{progress_text} (0/{len(batches)})")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {executor.submit(fetch_single_batch, batch, period): batch for batch in batches}
        for i, future in enumerate(as_completed(future_to_batch)):
            try:
                data, failed = future.result()
                if data is not None: all_data.append(data)
                if failed: failed_symbols.extend(failed)
            except Exception: failed_symbols.extend(future_to_batch[future])
            progress_bar.progress((i + 1) / len(batches), text=f"{progress_text} ({i+1}/{len(batches)})")
    progress_bar.empty()
    if failed_symbols: st.warning(f"Could not fetch data for: {', '.join(failed_symbols)}. They will be skipped.")
    return pd.concat(all_data, axis=1).dropna(how='all') if all_data else None

@st.cache_data(ttl=3600)
def get_stock_briefing(symbol):
    # (No changes to this function)
    try:
        company_name = yf.Ticker(f"{symbol}.NS").info.get('longName', symbol)
        query = f'"{company_name}" OR {symbol}'
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        return {"news": [a for a in articles.get('articles', []) if a.get('title')]}
    except Exception as e:
        st.sidebar.error(f"NewsAPI Error: {str(e)}", icon="📰")
        return {"news": None}

def style_dataframe(df):
    # (No changes to this function)
    def color_strategy(val):
        color = 'gray'
        if 'Buy' in str(val) or 'Breakout' in str(val): color = '#28a745'
        elif 'HOLD' in str(val) or 'WATCH' in str(val): color = '#ffc107'
        elif 'AVOID' in str(val): color = '#dc3545'
        return f'color: {color}; font-weight: bold;'
    def color_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0: color = '#28a745'
            elif val < 0: color = '#dc3545'
            else: color = 'gray'
            return f'color: {color};'
        return None
    styled_df = df.style
    if 'Strategy' in df.columns: styled_df = styled_df.map(color_strategy, subset=['Strategy'])
    if 'P&L' in df.columns: styled_df = styled_df.map(color_pnl, subset=['P&L'])
    if 'P&L %' in df.columns: styled_df = styled_df.map(color_pnl, subset=['P&L %'])
    return styled_df

# --- RUN PRE-LOADER ONCE AT STARTUP ---
if 'nse_stocks' not in st.session_state:
    with st.spinner("Pre-loading stock database for the first time... This is a one-time setup per session."):
        st.session_state.nse_stocks = _load_nse_stocks_cached(KITE_API_KEY)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Market Scanner", "🔍 Single Stock Research", "📈 My Portfolio"])

# --- TAB 1: MARKET SCANNER ---
with tab1:
    control_col, results_col = st.columns([1, 2.5])
    with control_col:
        with st.container(border=True):
            st.subheader("⚙️ Control Panel")
            scan_type = st.selectbox("Select Scan Type", ("Nifty 50", "Bank Nifty", "Nifty IT", "💥 Full Market"))
            capital = st.number_input("Enter Capital (INR)", 1000, 10000000, 100000, 1000)
            risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
            run_scan = st.button(f"🚀 Run {scan_type}", use_container_width=True)

    with results_col:
        if run_scan:
            stocks_to_analyze = []
            if "Full Market" in scan_type:
                # (Logic for pre-filtering remains the same)
                full_stock_list = st.session_state.nse_stocks
                if not full_stock_list:
                    st.error("Could not load any stocks. Scan cannot run."); st.stop()
                st.info(f"Stage 1: Downloading pre-filter data for {len(full_stock_list)} stocks...")
                pre_filter_data = fetch_market_data(full_stock_list, period="1mo", progress_text="Downloading pre-filter data...")
                if pre_filter_data is None:
                    st.error("Could not fetch pre-filter data. Scan cannot run."); st.stop()
                with st.spinner("Applying price & volume filters..."):
                    last_prices = pre_filter_data['Close'].iloc[-1]
                    avg_volumes = pre_filter_data['Volume'].rolling(window=20).mean().iloc[-1]
                    priced_stocks = last_prices[last_prices > MIN_PRICE].index
                    liquid_stocks = avg_volumes[avg_volumes > MIN_VOLUME].index
                    candidate_tickers = priced_stocks.intersection(liquid_stocks)
                    stocks_to_analyze = [ticker.replace('.NS', '') for ticker in candidate_tickers]
                st.success(f"Filter complete! Found {len(stocks_to_analyze)} stocks meeting your criteria.")
            else:
                stocks_to_analyze = get_watchlist(scan_type)
            
            if not stocks_to_analyze:
                st.warning("No stocks to analyze. Try adjusting filters.")
            else:
                st.info(f"Stage 2: Performing deep analysis on {len(stocks_to_analyze)} candidate stocks...")
                bulk_data = fetch_market_data(stocks_to_analyze, period="5y", progress_text="Downloading analysis data...")
                if bulk_data is not None and not bulk_data.empty:
                    results = []
                    analysis_progress = st.progress(0, text="Analyzing stocks...")
                    for i, s in enumerate(stocks_to_analyze):
                        res = analyze_stock(s, bulk_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                        if res: results.append(res)
                        analysis_progress.progress((i + 1) / len(stocks_to_analyze), text=f"Analyzing {s}...")
                    analysis_progress.empty()

                    if not results:
                        st.warning("Analysis complete, but no stocks had sufficient historical data.")
                    else:
                        ranked = pd.DataFrame(results)
                        # --- NEW: APPLY FUNDAMENTAL FILTERS ---
                        initial_count = len(ranked)
                        ranked.dropna(subset=['P/E', 'ROE %'], inplace=True) # Ensure we have data to filter on
                        ranked = ranked[ranked['P/E'] <= MAX_PE_RATIO]
                        ranked = ranked[ranked['ROE %'] >= MIN_ROE]
                        st.info(f"Applied fundamental filters: {initial_count} stocks -> {len(ranked)} stocks.")

                        strategy_order = ["🟢 Buy the Dip", "📈 Momentum Breakout", "🟡 WATCH", "🟡 HOLD", "🔴 AVOID", "N/A"]
                        ranked['Strategy'] = pd.Categorical(ranked['Strategy'], categories=strategy_order, ordered=True)
                        ranked.sort_values(by='Strategy', inplace=True)
                        
                        st.success(f"Analysis complete! Found {len(ranked[ranked['Strategy'].isin(['🟢 Buy the Dip', '📈 Momentum Breakout'])])} potential opportunities.")
                        with st.expander(f"🔍 Full Scan Results ({len(ranked)} stocks)", expanded=True):
                            # MODIFIED: Added fundamental columns
                            display_cols = ['Stock Name', 'Price', 'Strategy', 'Reason', 'RSI', 'P/E', 'ROE %', 'D/E']
                            st.dataframe(style_dataframe(ranked[display_cols]).format(precision=2), use_container_width=True, hide_index=True)
                        
                        # (The rest of the portfolio generation logic remains the same)
                        plan = []; cap = capital; risk_trade = capital * (risk_pct / 100)
                        for _, c in ranked.iterrows():
                            if not any(buy_sig in c['Strategy'] for buy_sig in ["Buy", "Breakout"]): continue
                            if c['Stop-Loss'] >= c['Price'] or c['Price'] <= 0: continue
                            risk_share = c['Price'] - c['Stop-Loss']
                            qty = min(math.floor(risk_trade / risk_share) if risk_share > 0 else 0, math.floor(cap / c['Price']))
                            if qty > 0: c['Qty'], c['Value'] = qty, qty * c['Price']; plan.append(c); cap -= c['Value']
                        if plan:
                            st.subheader("📈 Portfolio Strategy Summary")
                            inv = sum(p['Value'] for p in plan); profit = sum((p['Target'] - p['Entry']) * p['Qty'] for p in plan); loss = sum((p['Entry'] - p['Stop-Loss']) * p['Qty'] for p in plan)
                            m1,m2,m3,m4=st.columns(4); m1.metric("Capital Allocated", f"₹{inv:,.0f}", f"₹{cap:,.0f} Left"); m2.metric("Est. Profit", f"₹{profit:,.0f}"); m3.metric("Est. Loss", f"₹{loss:,.0f}"); m4.metric("R/R Ratio", f"{(profit/loss):.2f}:1" if loss > 0 else "∞")
                            st.subheader("📋 Generated Portfolio Plan"); df_plan=pd.DataFrame(plan)
                            st.dataframe(df_plan[['Stock Name', 'Qty', 'Price', 'Value', 'Entry', 'Target', 'Stop-Loss', 'Strategy']].style.format(precision=2), use_container_width=True, hide_index=True)
                            st.subheader("🔍 Analysis Zone")
                            sel_stock = st.selectbox("Select stock for analysis:", df_plan['Stock Name'].tolist())
                            if sel_stock:
                                info = next(item for item in results if item['Stock Name'] == sel_stock)
                                c1, c2 = st.columns(2); c1.plotly_chart(create_stock_chart(info['Chart Data'], info['Stock Name']), use_container_width=True)
                                with c2:
                                    news = get_stock_briefing(info['Stock Name'])
                                    if news and news.get('news'):
                                        for item in news['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                                    else: st.info("No recent news found.")
                        else: st.info("No actionable 'Buy' signals found. Try adjusting filters or check full results for stocks to 'WATCH'.")
                else: st.error("Data fetching for deep analysis failed.")
        else: st.info("⬅️ Adjust parameters and click 'Run'.")

# --- TAB 2: SINGLE STOCK RESEARCH ---
with tab2:
    # (Minor modification to display fundamental data)
    st.subheader("🔬 On-Demand Stock Analysis")
    db = st.session_state.nse_stocks
    if db:
        query = st.text_input("Enter NSE symbol:", placeholder="e.g., RELIANCE, SBIN").upper()
        if query:
            matches = [s for s in db if query in s]; sel = None
            if not matches: st.error(f"No match for '{query}'.")
            elif len(matches) == 1: sel = matches[0]
            else: sel = st.radio("Multiple matches found:", matches, horizontal=True)
            if sel:
                with st.spinner(f"Analyzing {sel}..."):
                    data = fetch_market_data([sel])
                if data is None or data.empty:
                    st.error(f"Could not fetch data for {sel}.")
                else:
                    plan = analyze_stock(sel, data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                    if plan:
                        st.subheader(f"📋 Actionable Trading Plan for {sel}")
                        df_plan = pd.DataFrame([plan])
                        # MODIFIED: Added fundamental columns to display
                        display_cols = ['Stock Name', 'Price', 'Strategy', 'Reason', 'RSI', 'P/E', 'ROE %', 'D/E', 'Entry', 'Target', 'Stop-Loss']
                        st.dataframe(df_plan[display_cols].style.format(precision=2), use_container_width=True, hide_index=True)
                        st.divider()
                        st.subheader(f"📰 News & Chart for {sel}")
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(create_stock_chart(plan['Chart Data'], sel), use_container_width=True)
                        with c2:
                            news = get_stock_briefing(sel)
                            if news and news.get('news'):
                                for item in news['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                            else: st.info("No recent news found.")
                    else: st.warning(f"Could not generate a plan for {sel}. (Insufficient historical data)")
    else: st.warning("Stock database could not be loaded. Search is disabled.")

# --- TAB 3: MY PORTFOLIO ---
with tab3:
    # (Minor modification to display fundamental data and enhance AI)
    st.subheader("📋 My Live Portfolio Tracker")
    stock_db = st.session_state.nse_stocks
    
    # (Add/Edit/Remove logic remains the same)
    with st.container(border=True):
        st.write("**Add a New Stock to Your Portfolio**")
        with st.form("add_stock_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            symbol = c1.selectbox("Select Stock", options=stock_db, index=None, placeholder="Type to search...")
            quantity = c2.number_input("Quantity", min_value=1, step=1)
            buy_price_str = c3.text_input("Buy Price (per share)", placeholder="e.g., 87.69")
            submitted = st.form_submit_button("Add Stock")

            if submitted and symbol and quantity > 0 and buy_price_str:
                try:
                    buy_price = float(buy_price_str)
                    if buy_price <= 0: st.error("Buy price must be a positive number.")
                    else:
                        if any(s['symbol'] == symbol for s in st.session_state.portfolio):
                            st.warning(f"{symbol} is already in your portfolio. Use 'Remove' and add again.")
                        else:
                            st.session_state.portfolio.append({"symbol": symbol, "quantity": quantity, "buy_price": buy_price})
                            save_portfolio_to_disk(st.session_state.portfolio)
                            if 'processed_portfolio' in st.session_state: del st.session_state['processed_portfolio']
                            st.success(f"Added {quantity} shares of {symbol}. Click Refresh to see it.")
                except ValueError: st.error("Invalid Buy Price. Please enter a valid number.")
    
    st.divider()

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add a stock to begin tracking.", icon="📈")
    else:
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            portfolio_symbols = [s['symbol'] for s in st.session_state.portfolio]
            with st.spinner("Fetching latest data..."):
                portfolio_data = fetch_market_data(portfolio_symbols, period="5y", progress_text="Fetching data...")
            processed_portfolio = []
            for stock in st.session_state.portfolio:
                symbol, qty, buy_price = stock['symbol'], stock['quantity'], stock['buy_price']
                buy_value = qty * buy_price
                processed_stock = { "Stock": symbol, "Qty": qty, "Buy Price": buy_price, "CMP": np.nan, "Buy Value": buy_value, "Current Value": np.nan, "P&L": np.nan, "P&L %": np.nan, "Strategy": "Data N/A", "P/E": np.nan, "ROE %": np.nan }
                if portfolio_data is not None and f"{symbol}.NS" in portfolio_data.columns.get_level_values(1):
                    try:
                        analysis_result = analyze_stock(symbol, portfolio_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                        if analysis_result:
                            current_price = analysis_result['Price']
                            current_value = qty * current_price
                            pnl = current_value - buy_value
                            pnl_pct = (pnl / buy_value) * 100 if buy_value > 0 else 0
                            processed_stock.update({ "CMP": current_price, "Current Value": current_value, "P&L": pnl, "P&L %": pnl_pct, "Strategy": analysis_result['Strategy'], "P/E": analysis_result['P/E'], "ROE %": analysis_result['ROE %'] })
                    except (KeyError, IndexError): pass 
                processed_portfolio.append(processed_stock)
            if processed_portfolio:
                st.session_state.processed_portfolio = pd.DataFrame(processed_portfolio)

        if 'processed_portfolio' in st.session_state and not st.session_state.processed_portfolio.empty:
            df_portfolio = st.session_state.processed_portfolio
            # (Summary metrics logic remains the same)
            total_investment = sum(s['quantity'] * s['buy_price'] for s in st.session_state.portfolio)
            calc_df = df_portfolio.dropna(subset=['P&L'])
            total_current_value = calc_df['Current Value'].sum()
            total_pnl = total_current_value - total_investment
            total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            
            st.subheader("📊 Portfolio Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Investment", f"₹{total_investment:,.2f}")
            m2.metric("Current Value", f"₹{total_current_value:,.2f}")
            m3.metric("Overall P&L", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            
            st.subheader("Holding Details")
            # MODIFIED: Added fundamental columns
            display_cols = ["Stock", "Qty", "Buy Price", "CMP", "Current Value", "P&L", "P&L %", "Strategy", "P/E", "ROE %"]
            st.dataframe(style_dataframe(df_portfolio[display_cols]).format(precision=2), use_container_width=True, hide_index=True)
            
            st.divider()

            # --- AI ADVISOR (ENHANCED WITH FUNDAMENTAL DATA) ---
            st.subheader("🤖 AI Portfolio Advisor")
            if not GEMINI_API_KEY:
                st.warning("Please add your Google Gemini API key to `.streamlit/secrets.toml` to enable the AI Advisor.", icon="🔑")
            else:
                prompt = st.text_input("Ask a question about your portfolio...", placeholder="e.g., Should I buy more ASHOKLEY? or Compare TATAMOTORS and M&M.")
                if prompt:
                    # MODIFIED: Added fundamental data to the prompt
                    portfolio_summary = df_portfolio[['Stock', 'Qty', 'Buy Price', 'CMP', 'P&L %', 'P/E', 'ROE %']].to_string()
                    full_prompt = f"""
                    You are an expert stock market analyst. A user has a portfolio with the following summary (including key fundamentals) and has asked a question.
                    Analyze the user's question in the context of their portfolio, the stock's fundamental health (P/E, ROE), and current market data.
                    Provide a balanced, data-driven recommendation. Do not give absolute financial advice, but rather present the pros, cons, and key factors to consider.

                    USER'S PORTFOLIO SUMMARY:
                    {portfolio_summary}

                    USER'S QUESTION:
                    "{prompt}"

                    Based on this, please provide your analysis:
                    """
                    try:
                        with st.spinner("🤖 AI is analyzing fundamentals and technicals..."):
                            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                            response = model.generate_content(full_prompt)
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"An error occurred while contacting the AI model: {e}")

            st.divider()
            # (Deeper Analysis and Edit/Remove sections remain the same)
            st.subheader("🔍 Deeper Analysis")
            available_stocks = df_portfolio.dropna(subset=['CMP'])['Stock'].tolist()
            selected_stock = st.selectbox("Select a stock from your portfolio to see its chart and news", options=available_stocks, index=None)
            if selected_stock:
                with st.spinner(f"Loading details for {selected_stock}..."):
                    stock_data = fetch_market_data([selected_stock])
                    if stock_data is not None and not stock_data.empty:
                        analysis_result = analyze_stock(selected_stock, stock_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                        c1, c2 = st.columns(2)
                        if analysis_result: c1.plotly_chart(create_stock_chart(analysis_result['Chart Data'], selected_stock), use_container_width=True)
                        else: c1.warning("Could not generate a chart.")
                        with c2:
                            st.write("**Recent News**")
                            news = get_stock_briefing(selected_stock)
                            if news and news.get('news'):
                                for item in news['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                            else: st.info("No recent news found.")
                    else: st.error(f"Could not fetch detailed data for {selected_stock}.")
        else:
            st.info("Click 'Refresh Live Data' to fetch prices and analysis.")

        with st.expander("✏️ Edit Stock Details"):
            if st.session_state.portfolio:
                stock_to_edit = st.selectbox("Select stock to edit", options=[s['symbol'] for s in st.session_state.portfolio], index=None, key="stock_editor")
                if stock_to_edit:
                    stock_data = next((s for s in st.session_state.portfolio if s['symbol'] == stock_to_edit), None)
                    if stock_data:
                        col1, col2 = st.columns(2)
                        new_quantity = col1.number_input("Correct Quantity", min_value=1, step=1, value=stock_data['quantity'], key=f"qty_edit_{stock_to_edit}")
                        new_buy_price_str = col2.text_input("Correct Buy Price", value=f"{stock_data['buy_price']:.2f}", key=f"price_edit_{stock_to_edit}")
                        if st.button("Update Details", key="update_details_button"):
                            try:
                                new_buy_price = float(new_buy_price_str)
                                if new_buy_price <= 0: st.error("Buy price must be a positive number.")
                                else:
                                    for stock in st.session_state.portfolio:
                                        if stock['symbol'] == stock_to_edit:
                                            stock['quantity'] = new_quantity
                                            stock['buy_price'] = new_buy_price
                                            break
                                    save_portfolio_to_disk(st.session_state.portfolio)
                                    if 'processed_portfolio' in st.session_state: del st.session_state['processed_portfolio']
                                    st.success(f"Details for {stock_to_edit} updated. Click 'Refresh' to see new calculations.")
                                    st.rerun()
                            except ValueError: st.error("Please enter a valid number for the buy price.")
        
        with st.expander("🗑️ Remove a stock from portfolio"):
            if st.session_state.portfolio:
                stock_to_remove = st.selectbox("Select stock to remove", options=[s['symbol'] for s in st.session_state.portfolio], index=None, key="stock_remover")
                if st.button("Remove Selected Stock", disabled=(not stock_to_remove)):
                    st.session_state.portfolio = [s for s in st.session_state.portfolio if s['symbol'] != stock_to_remove]
                    save_portfolio_to_disk(st.session_state.portfolio)
                    if 'processed_portfolio' in st.session_state:
                        df = st.session_state.processed_portfolio
                        st.session_state.processed_portfolio = df[df.Stock != stock_to_remove]
                    st.rerun()

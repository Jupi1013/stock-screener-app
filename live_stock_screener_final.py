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

st.sidebar.subheader("Fundamental Filters")
MAX_PE_RATIO = st.sidebar.number_input("Maximum P/E Ratio", 1, 200, 50, 5)
MIN_ROE = st.sidebar.number_input("Minimum Return on Equity (ROE %)", -50, 100, 15, 1)

st.sidebar.subheader("General Filters")
MIN_PRICE = st.sidebar.slider("Minimum Stock Price (₹)", 10, 500, 50, 10)
MIN_VOLUME = st.sidebar.slider("Minimum 20-Day Avg Volume", 10000, 1000000, 100000, 10000)

# --- UTILITY FUNCTIONS ---
def get_watchlist(index_name):
    watchlists = {
        "Bank Nifty": ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'AUBANK', 'BANDHANBNK'],
        "Nifty 50": ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR', 'BHARTIARTL', 'ITC', 'LTIM', 'SBIN', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'TATAMOTORS', 'ADANIENT', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'TATASTEEL', 'AXISBANK', 'ONGC', 'NTPC', 'BAJAJFINSV', 'ADANIPORTS', 'NESTLEIND', 'COALINDIA', 'WIPRO', 'POWERGRID', 'M&M', 'GRASIM', 'JSWSTEEL', 'HINDALCO', 'ULTRACEMCO', 'EICHERMOT', 'DRREDDY', 'CIPLA', 'INDUSINDBK', 'BRITANNIA', 'HEROMOTOCO', 'APOLLOHOSP', 'DIVISLAB', 'BPCL', 'SHREECEM', 'UPL', 'SBILIFE', 'TECHM', 'BAJAJ-AUTO', 'ADANIGREEN', 'TATACONSUM'],
        "Nifty IT": ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'LTIM', 'TECHM', 'PERSISTENT', 'OFSS', 'MPHASIS', 'COFORGE']
    }
    return watchlists.get(index_name, [])

def get_fallback_stocks():
    return ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR', 'BHARTIARTL', 'ITC', 'LTIM', 'SBIN', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'TATAMOTORS', 'ADANIENT', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'TATASTEEL', 'AXISBANK', 'ONGC', 'NTPC', 'BAJAJFINSV', 'ADANIPORTS', 'NESTLEIND', 'COALINDIA', 'WIPRO', 'POWERGRID', 'M&M', 'GRASIM', 'JSWSTEEL', 'HINDALCO', 'ULTRACEMCO', 'EICHERMOT', 'DRREDDY', 'CIPLA', 'INDUSINDBK', 'BRITANNIA', 'HEROMOTOCO', 'APOLLOHOSP', 'DIVISLAB', 'BPCL', 'SHREECEM', 'UPL', 'SBILIFE', 'TECHM', 'BAJAJ-AUTO', 'ADANIGREEN', 'TATACONSUM', 'ZOMATO', 'PAYTM', 'POLICYBZR', 'NYKAA', 'VEDL', 'ITC', 'IOC', 'BHEL', 'GAIL', 'SAIL', 'DLF']

@st.cache_data(ttl=86400)
def _load_nse_stocks_cached(api_key):
    try:
        kite = KiteConnect(api_key=api_key)
        instruments = kite.instruments("NSE")
        
        equity_symbols = []
        exclude_name_keywords = ['ETF', 'BEES', 'LIQUID', 'GOLD', 'SILVER', 'GILT', 'NIFTY', 'SENSEX', 'BOND']

        for inst in instruments:
            if not (isinstance(inst, dict) and inst.get('instrument_type') == 'EQ' and inst.get('exchange') == 'NSE'):
                continue

            symbol = inst.get('tradingsymbol', '')
            name = inst.get('name', '').upper()
            is_excluded = any(keyword in name for keyword in exclude_name_keywords)
            
            if symbol and symbol.isalpha() and not is_excluded:
                 equity_symbols.append(symbol)
        
        if not equity_symbols:
            st.sidebar.warning("Could not filter any valid equity symbols, using fallback list.")
            return get_fallback_stocks()
        
        return sorted(list(set(equity_symbols)))
    except Exception:
        st.sidebar.warning(f"Could not connect to Zerodha. Using a fallback list.")
        return get_fallback_stocks()

def create_stock_chart(df, stock_symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if 'SMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange', dash='dash')))
    if 'SMA200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'{stock_symbol} Candlestick Chart', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=400, margin=dict(l=20, r=20, t=40, b=20), xaxis_rangeslider_visible=False)
    return fig

@st.cache_data(ttl=3600)
def analyze_stock(stock_symbol, bulk_data_frame, atr_multiplier, rsi_overbought, volume_multiplier):
    try:
        ticker_str = f"{stock_symbol.upper()}.NS"
        df = bulk_data_frame.xs(ticker_str, level=1, axis=1).copy() if isinstance(bulk_data_frame.columns, pd.MultiIndex) else bulk_data_frame.copy()
        df.dropna(how='all', inplace=True)
        if df.empty or len(df) < 200: return None
        
        has_volume = 'Volume' in df.columns and pd.to_numeric(df['Volume'], errors='coerce').notna().any()
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
        
        pe_ratio, pb_ratio, roe, debt_to_equity = np.nan, np.nan, np.nan, np.nan
        try:
            info = yf.Ticker(ticker_str).info
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            debt_to_equity = info.get('debtToEquity', np.nan)
        except Exception: pass 
        df_30 = chart_df.tail(30)
        return {
            "Stock Name": stock_symbol, "Price": price, "Strategy": strategy, "RSI": last['RSI'],
            "Entry": df_30['Low'].min() if strategy == "🟢 Buy the Dip" else price,
            "Target": df_30['High'].max(), "Stop-Loss": price - (atr_multiplier * last['ATR']),
            "Suggested Hold (Days)": abs((df_30['High'].idxmax() - df_30['Low'].idxmin()).days) or 1,
            "Vol Ratio": volume_ratio, "Chart Data": chart_df, "Reason": reason,
            "P/E": pe_ratio, "P/B": pb_ratio, "ROE %": roe * 100 if pd.notna(roe) else np.nan, "D/E": debt_to_equity
        }
    except (KeyError, IndexError, TypeError): return None

def fetch_single_batch(batch, period):
    tickers = [f"{s.upper()}.NS" for s in batch]
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True, threads=False)
    if data.empty: return None, batch
    failed = [t.replace('.NS', '') for t in tickers if t.upper() not in data.columns.get_level_values(1).unique()]
    return data, failed

def fetch_market_data(symbols, period="5y"):
    if not symbols: return None
    all_data, failed_symbols = [], []
    batches = [symbols[i:i+100] for i in range(0, len(symbols), 100)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {executor.submit(fetch_single_batch, batch, period): batch for batch in batches}
        for future in as_completed(future_to_batch):
            try:
                data, failed = future.result()
                if data is not None: all_data.append(data)
                if failed: failed_symbols.extend(failed)
            except Exception: failed_symbols.extend(future_to_batch[future])
            
    if failed_symbols:
        if len(failed_symbols) > 15: st.sidebar.warning(f"Could not fetch data for {len(failed_symbols)} symbols. They were skipped.")
        elif failed_symbols: st.sidebar.warning(f"Could not fetch data for: {', '.join(failed_symbols)}. They were skipped.")
    return pd.concat(all_data, axis=1).dropna(how='all') if all_data else None

@st.cache_data(ttl=3600)
def get_stock_briefing(symbol):
    try:
        company_name = yf.Ticker(f"{symbol}.NS").info.get('longName', symbol)
        query = f'"{company_name}" OR {symbol}'
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        return {"news": [a for a in articles.get('articles', []) if a.get('title')]}
    except Exception as e:
        st.sidebar.error(f"NewsAPI Error: {str(e)}", icon="📰")
        return {"news": None}

def get_recommendation_stamp(symbol, analysis_data, news_articles):
    if not GEMINI_API_KEY:
        return "**Verdict:** Analysis disabled\n**Justification:** Please set your Gemini API key in the secrets.toml file to enable this feature."
    if not news_articles or not news_articles.get('news'):
        news_summary = "No recent news was found."
    else:
        headlines = [item['title'] for item in news_articles['news'][:5]]
        news_summary = "\n".join(f"- {h}" for h in headlines)
    prompt_template = f"""
    Analyze the following real-time data for the Indian stock '{symbol}' and provide a very concise recommendation stamp.
    **Technical Snapshot:**
    - **System-Generated Strategy:** {analysis_data.get('Strategy', 'N/A')}
    - **Reasoning:** {analysis_data.get('Reason', 'N/A')}
    - **Current Price:** {analysis_data.get('Price', 0):.2f} INR
    - **RSI (14):** {analysis_data.get('RSI', 0):.2f}
    **Recent News Headlines:**
    {news_summary}
    **Your Task:**
    Based *only* on the data provided above, give a one-line verdict. Then, provide a brief 2-3 sentence justification. Format your response exactly as follows:
    **Verdict:** [Your one-line verdict]
    **Justification:** [Your brief explanation]
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"**Verdict:** Error\n**Justification:** Could not get stamp due to an error: {e}"

def style_dataframe(df):
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
    with st.spinner("Pre-loading stock database for the first time..."):
        st.session_state.nse_stocks = _load_nse_stocks_cached(KITE_API_KEY)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Scanner", "🔍 Single Stock Research", "📈 My Portfolio", "💡 Opportunities"])

# --- TAB 1: MARKET SCANNER ---
with tab1:
    control_col, results_col = st.columns([1, 2.5])
    with control_col:
        with st.container(border=True):
            st.subheader("⚙️ Control Panel")
            scan_type = st.selectbox("Select Scan Type", ("Nifty 50", "Bank Nifty", "Nifty IT", "💥 Full Market"))
            capital = st.number_input("Enter Capital (INR)", 1000, 10000000, 100000, 1000)
            risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
            
            if st.button(f"🚀 Run {scan_type}", use_container_width=True):
                st.session_state.scan_params = {"scan_type": scan_type, "capital": capital, "risk_pct": risk_pct}
                st.session_state.run_scan_flag = True
                if 'scan_results' in st.session_state: del st.session_state.scan_results

    with results_col:
        if st.session_state.get('run_scan_flag', False):
            with st.spinner("Running scan..."):
                params = st.session_state.scan_params
                scan_type, capital, risk_pct = params["scan_type"], params["capital"], params["risk_pct"]
                
                stocks_to_analyze = []
                if scan_type == "💥 Full Market":
                    stocks_to_analyze = st.session_state.nse_stocks
                else:
                    stocks_to_analyze = get_watchlist(scan_type)
                
                if stocks_to_analyze:
                    bulk_data = fetch_market_data(stocks_to_analyze)
                    if bulk_data is not None and not bulk_data.empty:
                        results = [res for s in stocks_to_analyze if (res := analyze_stock(s, bulk_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)) is not None]
                        st.session_state.scan_results = {"results": results, "capital": capital, "risk_pct": risk_pct}
            st.session_state.run_scan_flag = False

        if 'scan_results' in st.session_state:
            results_data = st.session_state.scan_results
            if results_data:
                results = results_data["results"]
                capital = results_data["capital"]
                risk_pct = results_data["risk_pct"]

                ranked = pd.DataFrame(results)
                # ... [Rest of the display logic for Tab 1]
            else:
                st.info("Scan complete. No stocks matched the criteria.")

        else:
            st.info("⬅️ Adjust parameters and click 'Run'.")


# --- TAB 2: SINGLE STOCK RESEARCH ---
with tab2:
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
                with st.spinner(f"Analyzing {sel}..."): data = fetch_market_data([sel])
                if data is None or data.empty: st.error(f"Could not fetch data for {sel}.")
                else:
                    plan = analyze_stock(sel, data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                    if plan:
                        st.subheader(f"📋 Actionable Trading Plan for {sel}")
                        st.dataframe(pd.DataFrame([plan])[['Stock Name', 'Price', 'Strategy', 'Reason', 'RSI', 'P/E', 'ROE %', 'Entry', 'Target', 'Stop-Loss']].style.format(precision=2), use_container_width=True, hide_index=True)
                        st.divider()
                        st.subheader(f"📰 News & Chart for {sel}")
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(create_stock_chart(plan['Chart Data'], sel), use_container_width=True)
                        with c2:
                            st.write("**Recommendation Stamp**"); news = get_stock_briefing(sel)
                            with st.spinner("Analyzing..."): st.info(get_recommendation_stamp(sel, plan, news))
                            st.write("**Recent News**")
                            if news and news.get('news'):
                                for item in news['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                            else: st.info("No recent news found.")
                    else: st.warning(f"Could not generate a plan for {sel}. (Insufficient historical data)")
    else: st.warning("Stock database could not be loaded. Search is disabled.")

# --- TAB 3: MY PORTFOLIO ---
with tab3:
    st.subheader("📋 My Live Portfolio Tracker")
    stock_db = st.session_state.nse_stocks
    
    with st.container(border=True):
        st.write("**Add a New Stock to Your Portfolio**")
        with st.form("add_stock_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            symbol = c1.selectbox("Select Stock", options=stock_db, index=None, placeholder="Type to search...")
            quantity = c2.number_input("Quantity", min_value=1, step=1)
            buy_price_str = c3.text_input("Buy Price (per share)", placeholder="e.g., 87.69")
            if st.form_submit_button("Add Stock") and symbol and quantity > 0 and buy_price_str:
                try:
                    buy_price = float(buy_price_str)
                    if not any(s['symbol'] == symbol for s in st.session_state.portfolio):
                        st.session_state.portfolio.append({"symbol": symbol, "quantity": quantity, "buy_price": buy_price})
                        save_portfolio_to_disk(st.session_state.portfolio)
                        if 'processed_portfolio' in st.session_state: del st.session_state['processed_portfolio']
                    else: st.warning(f"{symbol} is already in your portfolio.")
                except ValueError: st.error("Invalid Buy Price.")
    
    st.divider()

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add a stock to begin tracking.", icon="📈")
    else:
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.session_state.refresh_portfolio = True

        if st.session_state.get('refresh_portfolio', False):
             with st.spinner("Fetching latest data..."):
                portfolio_symbols = [s['symbol'] for s in st.session_state.portfolio]
                portfolio_data = fetch_market_data(portfolio_symbols, period="5y")
                processed_portfolio = []
                if portfolio_data is not None:
                    for stock in st.session_state.portfolio:
                        symbol, qty, buy_price = stock['symbol'], stock['quantity'], stock['buy_price']
                        analysis_result = analyze_stock(symbol, portfolio_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)
                        if analysis_result:
                            current_price = analysis_result['Price']
                            buy_value = qty * buy_price
                            current_value = qty * current_price
                            pnl = current_value - buy_value
                            analysis_result.update({"Qty": qty, "Buy Price": buy_price, "Buy Value": buy_value, "Current Value": current_value, "P&L": pnl, "P&L %": (pnl / buy_value) * 100 if buy_value > 0 else 0})
                            processed_portfolio.append(analysis_result)
                st.session_state.processed_portfolio = pd.DataFrame(processed_portfolio)
                st.session_state.refresh_portfolio = False

        if 'processed_portfolio' in st.session_state and not st.session_state.processed_portfolio.empty:
            df_portfolio = st.session_state.processed_portfolio
            total_investment = df_portfolio['Buy Value'].sum()
            total_current_value = df_portfolio['Current Value'].sum()
            total_pnl = df_portfolio['P&L'].sum()
            total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            
            st.subheader("📊 Portfolio Summary")
            m1, m2, m3 = st.columns(3); m1.metric("Total Investment", f"₹{total_investment:,.2f}"); m2.metric("Current Value", f"₹{total_current_value:,.2f}"); m3.metric("Overall P&L", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            
            st.subheader("Holding Details")
            st.dataframe(style_dataframe(df_portfolio[["Stock Name", "Qty", "Buy Price", "Price", "Current Value", "P&L", "P&L %", "Strategy", "P/E", "ROE %"]]).format(precision=2), use_container_width=True, hide_index=True, column_config={"Stock Name": "Stock", "Price": "CMP"})
            
            # ... [Rest of Tab 3 logic remains unchanged] ...

# --- TAB 4: OPPORTUNITIES ---
with tab4:
    st.subheader("💡 Vetted Market Opportunities")
    st.markdown("This tool searches the entire market using a **built-in, balanced strategy** to find high-quality opportunities.")
    DEFAULT_STRATEGY_PARAMS = {"MIN_PRICE": 100, "MIN_VOLUME": 500000, "PE_MAX": 40, "ROE_MIN": 15, "RSI_MAX": 65, "ATR_MULTIPLIER": 2.0}
    
    if st.button("🚀 Find Market Opportunities", use_container_width=True):
        st.session_state.run_opp_scan = True
        if 'recommendation_results' in st.session_state: del st.session_state['recommendation_results']
    
    if st.session_state.get('run_opp_scan', False):
        with st.spinner("Performing multi-stage market analysis..."):
            full_stock_list = st.session_state.nse_stocks
            batches = [full_stock_list[i:i+100] for i in range(0, len(full_stock_list), 100)]
            technically_strong_stocks = []
            prog = st.progress(0, text="Screening batches...")
            for i, batch in enumerate(batches):
                screening_data = fetch_market_data(batch, period="1y")
                prog.progress((i + 1) / len(batches), text=f"Screening batch {i+1} of {len(batches)}...")
                if screening_data is None: continue
                for symbol in batch:
                    try:
                        ticker_str = f"{symbol.upper()}.NS"
                        if ticker_str in screening_data.columns.get_level_values(1):
                            df = screening_data.xs(ticker_str, level=1, axis=1).copy()
                            if len(df) > 200 and df['Close'].iloc[-1] > DEFAULT_STRATEGY_PARAMS['MIN_PRICE'] and df['Volume'].rolling(window=20).mean().iloc[-1] > DEFAULT_STRATEGY_PARAMS['MIN_VOLUME']:
                                df.ta.sma(length=50, append=True, col_names=("SMA50",)); df.ta.sma(length=200, append=True, col_names=("SMA200",))
                                df.ta.rsi(length=14, append=True, col_names=("RSI",)); df.ta.atr(length=14, append=True, col_names=("ATR",))
                                df.dropna(inplace=True)
                                if not df.empty and (df.iloc[-1]['Close'] > df.iloc[-1]['SMA50']) and (df.iloc[-1]['Close'] > df.iloc[-1]['SMA200']) and (df.iloc[-1]['RSI'] < DEFAULT_STRATEGY_PARAMS['RSI_MAX']):
                                    technically_strong_stocks.append({'symbol': symbol, 'chart_data': df, 'full_df': screening_data.xs(ticker_str, level=1, axis=1).copy()})
                    except: continue
            prog.empty()
            
            final_recommendations = []
            if technically_strong_stocks:
                prog = st.progress(0, text="Vetting candidates...")
                for i, stock in enumerate(technically_strong_stocks):
                    try:
                        info = yf.Ticker(f"{stock['symbol']}.NS").info
                        pe_ratio, roe = info.get('trailingPE'), info.get('returnOnEquity')
                        if pe_ratio is not None and roe is not None and (pe_ratio > 0 and pe_ratio < DEFAULT_STRATEGY_PARAMS['PE_MAX']) and ((roe * 100) > DEFAULT_STRATEGY_PARAMS['ROE_MIN']):
                            last_row = stock['chart_data'].iloc[-1]; price = last_row['Close']; df_30 = stock['full_df'].tail(30)
                            hold_days = abs((df_30['High'].idxmax() - df_30['Low'].idxmin()).days) or 1
                            final_recommendations.append({
                                "Stock Name": stock['symbol'], "Price": price, "Strategy": "📈 Vetted Buy Signal", "Entry": price,
                                "Target": df_30['High'].max(), "Stop-Loss": price - (DEFAULT_STRATEGY_PARAMS['ATR_MULTIPLIER'] * last_row['ATR']),
                                "Suggested Hold (Days)": hold_days, "RSI": last_row['RSI'], "P/E": pe_ratio, "ROE %": roe * 100, "Chart Data": stock['full_df']
                            })
                    except: continue
                    prog.progress((i + 1) / len(technically_strong_stocks), text=f"Vetting {stock['symbol']}...")
                prog.empty()

            st.session_state.recommendation_results = pd.DataFrame(final_recommendations)
        st.session_state.run_opp_scan = False
    
    if 'recommendation_results' in st.session_state:
        df_rec = st.session_state.recommendation_results
        if not df_rec.empty:
            st.success(f"Scan complete! Found {len(df_rec)} high-quality opportunities.")
            display_cols = ["Stock Name", "Price", "Strategy", "Entry", "Target", "Stop-Loss", "Suggested Hold (Days)", "P/E", "ROE %"]
            st.dataframe(style_dataframe(df_rec[display_cols]).format(precision=2), use_container_width=True, hide_index=True, column_config={"Stock Name": "Stock", "Price": "CMP", "Stop-Loss": "Stop Loss", "Suggested Hold (Days)": st.column_config.NumberColumn("Hold (Days)", format="%d")})
            
            st.divider()
            st.subheader("🔍 Analysis Zone")
            selected_stock = st.selectbox("Select a recommended stock for detailed analysis:", options=df_rec['Stock Name'].tolist(), index=None, key="rec_stock_selector")
            if selected_stock:
                info = df_rec[df_rec['Stock Name'] == selected_stock].to_dict('records')[0]
                c1, c2 = st.columns(2)
                c1.plotly_chart(create_stock_chart(info['Chart Data'], info['Stock Name']), use_container_width=True)
                with c2:
                    st.write("**Recommendation Stamp**"); news = get_stock_briefing(info['Stock Name'])
                    with st.spinner("Analyzing..."): st.info(get_recommendation_stamp(selected_stock, info, news))
                    st.write("**Recent News**")
                    if news and news.get('news'):
                        for item in news['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                    else: st.info("No recent news found.")
        else:
            st.info("Scan complete. No stocks matched the built-in quality criteria today.")
    else:
        st.info("Click the button above to scan the market for recommended stocks.")

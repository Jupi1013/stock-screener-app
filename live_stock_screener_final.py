# In file: ~/Documents/My Python Project/live_stock_screener_final.py

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
st.set_page_config(page_title="Rohan's Strategy", page_icon="📈", layout="wide")

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
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        st.info("First-time setup: Downloading sentiment analysis model...")
        nltk.download('vader_lexicon')

download_nltk_data()

# --- Functions to Save and Load Portfolio from a File ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(SCRIPT_DIR, "portfolio.json")
SIP_PORTFOLIO_FILE = os.path.join(SCRIPT_DIR, "sip_portfolio.json")

# --- FIX STARTS HERE: Define FOUR distinct parts of your SIP plan ---
MONTHLY_MUTUAL_FUNDS = [
    {"symbol": "ICICINXT50.NS", "type": "Mutual Fund", "name": "ICICI Pru Nifty Next 50", "amount": 1250},
    {"symbol": "SETFNIF50.NS", "type": "Mutual Fund", "name": "SBI Flexicap (Proxy)", "amount": 1250},
]
MONTHLY_STOCKS = [
    {"symbol": "MAXHEALTH.NS", "type": "Stock", "name": "Max Healthcare", "amount": 1250},
    {"symbol": "INDHOTEL.NS", "type": "Stock", "name": "Indian Hotels", "amount": 1250},
    {"symbol": "CYIENTDLM.NS", "type": "Stock", "name": "Cyient DLM", "amount": 1200},
    {"symbol": "KARURVYSYA.NS", "type": "Stock", "name": "Karur Vysya Bank", "amount": 1200},
    {"symbol": "IRB.NS", "type": "Stock", "name": "IRB Infra", "amount": 1100},
    {"symbol": "IDFCFIRSTB.NS", "type": "Stock", "name": "IDFC First Bank", "amount": 1030},
]
MONTHLY_ETFS = [
    {"symbol": "GOLDBEES.NS", "type": "ETF", "name": "Gold ETF (Monthly)", "amount": 750},
    {"symbol": "SILVERBEES.NS", "type": "ETF", "name": "Silver ETF (Monthly)", "amount": 750},
]
WEEKLY_ETFS = [
    {"symbol": "GOLDBEES.NS", "type": "ETF", "name": "Gold ETF (Weekly)", "amount": 500},
    {"symbol": "SILVERBEES.NS", "type": "ETF", "name": "Silver ETF (Weekly)", "amount": 500},
]
# --- FIX ENDS HERE ---

def load_portfolio_from_disk():
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        if os.path.getsize(PORTFOLIO_FILE) == 0:
            st.sidebar.warning("`portfolio.json` was found but is empty.", icon="⚠️")
            return []
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.sidebar.error("Could not read `portfolio.json`. File may be corrupt.", icon="🚨")
        return []
    except Exception as e:
        st.sidebar.error(f"Error loading portfolio: {e}", icon="🚨")
        return []

def save_portfolio_to_disk(portfolio_data):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio_data, f, indent=4)

def load_sip_portfolio():
    default_state = {"last_monthly_sip_date": None, "last_weekly_sip_date": None, "holdings": []}
    if not os.path.exists(SIP_PORTFOLIO_FILE):
        return default_state
    try:
        if os.path.getsize(SIP_PORTFOLIO_FILE) == 0:
            st.sidebar.warning("Found `sip_portfolio.json`, but it is empty.", icon="⚠️")
            return default_state
        with open(SIP_PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            if 'holdings' not in data:
                st.sidebar.error("`sip_portfolio.json` is missing the 'holdings' key.", icon="🚨")
                return default_state
            return data
    except json.JSONDecodeError:
        st.sidebar.error("Could not read `sip_portfolio.json`. File may be corrupt.", icon="🚨")
        return default_state
    except Exception as e:
        st.sidebar.error(f"Error loading SIP portfolio: {e}", icon="🚨")
        return default_state

def save_sip_portfolio(data):
    with open(SIP_PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- INITIALIZE SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_disk()
if 'sip_portfolio_state' not in st.session_state:
    st.session_state.sip_portfolio_state = load_sip_portfolio()

# --- Header & Sidebar ---
st.title(" Rohan's Strategy ")
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

@st.cache_data
def get_stock_sector(symbol):
    """Fetches and caches the sector for a given stock symbol."""
    try:
        return yf.Ticker(f"{symbol}.NS").info.get('sector', 'Unknown')
    except Exception:
        return "Unknown"

@st.cache_data
def simulate_and_update_sip():
    sip_state = load_sip_portfolio()
    today = pd.to_datetime(date.today())
    updated = False

    # Process Monthly SIPs
    last_monthly_date_str = sip_state.get('last_monthly_sip_date')
    if last_monthly_date_str:
        last_monthly_date = pd.to_datetime(last_monthly_date_str)
        next_monthly_date = (last_monthly_date + pd.DateOffset(months=1)).replace(day=1)
        if today >= next_monthly_date:
            monthly_plan = MONTHLY_MUTUAL_FUNDS + MONTHLY_STOCKS + MONTHLY_ETFS
            all_monthly_symbols = [item['symbol'] for item in monthly_plan]
            hist_data = yf.download(all_monthly_symbols, start=next_monthly_date, end=today, progress=False)['Open']
            while next_monthly_date <= today:
                try:
                    price_series = hist_data.loc[next_monthly_date.strftime('%Y-%m-%d'):].iloc
                except IndexError: break
                for item in monthly_plan:
                    pass
                sip_state['last_monthly_sip_date'] = next_monthly_date.strftime('%Y-%m-%d')
                next_monthly_date += pd.DateOffset(months=1)
                updated = True

    # Process Weekly SIPs
    last_weekly_date_str = sip_state.get('last_weekly_sip_date')
    if last_weekly_date_str:
        last_weekly_date = pd.to_datetime(last_weekly_date_str)
        next_weekly_date = last_weekly_date + pd.DateOffset(weeks=1)
        if today >= next_weekly_date:
            hist_data = yf.download([item['symbol'] for item in WEEKLY_ETFS], start=next_weekly_date, end=today, progress=False)['Open']
            while next_weekly_date <= today:
                pass
                sip_state['last_weekly_sip_date'] = next_weekly_date.strftime('%Y-%m-%d')
                next_weekly_date += pd.DateOffset(weeks=1)
                updated = True

    if updated:
        save_sip_portfolio(sip_state)
    return sip_state, updated

def generate_portfolio_conclusion(df_portfolio):
    """Analyzes the entire portfolio and returns an AI-generated conclusion."""
    if not GEMINI_API_KEY:
        return """
        ### AI Analysis Disabled
        **Please set your `GEMINI_API_KEY` in the Streamlit secrets (`.streamlit/secrets.toml`) to enable this feature.**
        """

    try:
        df_for_prompt = df_portfolio[[
            "Stock Name", "Buy Value", "Current Value", "P&L", "P&L %", "Strategy", "P/E", "ROE %"
        ]].copy()
        
        for col in ["Buy Value", "Current Value", "P&L", "P&L %", "P/E", "ROE %"]:
            df_for_prompt[col] = df_for_prompt[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
        
        header = "| " + " | ".join(df_for_prompt.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df_for_prompt.columns)) + " |"
        rows = []
        for _, row in df_for_prompt.iterrows():
            rows.append("| " + " | ".join(str(x) for x in row.values) + " |")
        portfolio_summary_md = "\n".join([header, separator] + rows)

        total_investment = df_portfolio['Buy Value'].sum()
        total_current_value = df_portfolio['Current Value'].sum()
        total_pnl = df_portfolio['P&L'].sum()
        total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

        prompt = f"""
        You are a professional portfolio analyst reviewing a retail investor's stock portfolio. The data is from an Indian market context.
        **Portfolio Summary:**
        - **Total Investment:** ₹{total_investment:,.2f}
        - **Current Value:** ₹{total_current_value:,.2f}
        - **Overall P&L:** ₹{total_pnl:,.2f} ({total_pnl_pct:.2f}%)
        **Detailed Holdings:**
        {portfolio_summary_md}
        **Your Task:**
        Provide a concise, balanced conclusion of the portfolio's health in Markdown format with the following exact sections:
        ### Overall Health
        *A one or two-sentence summary of the portfolio's current state.*
        ### Strengths
        *Use bullet points to list 2-3 positive aspects. Consider profitable positions, diversification, or fundamentally strong companies (high ROE, reasonable P/E).*
        ### Potential Risks
        *Use bullet points to list 2-3 potential concerns. Consider over-concentration, stocks with high P/E ratios, negative ROE, or technical signals like "AVOID" or "HOLD".*
        ### Actionable Insights
        *Provide one or two clear, actionable suggestions for the investor to consider.*
        """

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"""
        ### 🤖 AI Analysis Failed
        **An error occurred while generating the conclusion.**
        This could be due to an invalid API key, network issues, or API restrictions.
        **Error Details:**
        ```
        {type(e).__name__}: {str(e)}
        ```
        Please check your `.streamlit/secrets.toml` file for a valid `GEMINI_API_KEY` and ensure you have an active internet connection, then try again.
        """

if 'nse_stocks' not in st.session_state:
    with st.spinner("Pre-loading stock database for the first time..."):
        st.session_state.nse_stocks = _load_nse_stocks_cached(KITE_API_KEY)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Market Scanner", "🔍 Single Stock Research", "📈 My Portfolio", "💡 Opportunities", "🚀 My SIP Dashboard"])

with tab1:
    st.subheader("📊 Market Scanner")
    control_col, results_col = st.columns([1, 2.5])

    with control_col:
        with st.container(border=True):
            st.subheader("⚙️ Control Panel")
            scan_type = st.selectbox("Select Scan Type", ("Nifty 50", "Bank Nifty", "Nifty IT", "💥 Full Market"), key="scanner_scan_type")
            capital = st.number_input("Enter Capital (INR)", 1000, 10000000, 100000, 1000, key="scanner_capital")
            risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5, key="scanner_risk_pct")

            if st.button(f"🚀 Run {scan_type}", use_container_width=True):
                st.session_state.run_scan_flag = True
                st.session_state.scan_params = {"scan_type": scan_type, "capital": capital, "risk_pct": risk_pct}
                if 'scan_results' in st.session_state:
                    del st.session_state.scan_results

    if 'run_scan_flag' in st.session_state and st.session_state.run_scan_flag:
        with st.spinner("Running scan..."):
            params = st.session_state.scan_params
            scan_type, capital, risk_pct = params["scan_type"], params["capital"], params["risk_pct"]
            stocks_to_analyze = get_watchlist(scan_type) if scan_type != "💥 Full Market" else st.session_state.nse_stocks

            if stocks_to_analyze:
                bulk_data = fetch_market_data(stocks_to_analyze)
                if bulk_data is not None and not bulk_data.empty:
                    results = [res for s in stocks_to_analyze if (res := analyze_stock(s, bulk_data, ATR_MULTIPLIER, RSI_OVERBOUGHT, VOLUME_MULTIPLIER)) is not None]
                    st.session_state.scan_results = {"results": results, "capital": capital, "risk_pct": risk_pct}
                else:
                    st.session_state.scan_results = {"results": [], "capital": capital, "risk_pct": risk_pct}
            else:
                st.session_state.scan_results = {"results": [], "capital": capital, "risk_pct": risk_pct}
        st.session_state.run_scan_flag = False
        st.rerun()

    with results_col:
        if 'scan_results' in st.session_state:
            results_data = st.session_state.scan_results
            if results_data and results_data.get("results"):
                ranked = pd.DataFrame(results_data["results"])
                ranked_filtered = ranked[
                    (ranked['Price'] >= MIN_PRICE) &
                    (ranked['P/E'].fillna(MAX_PE_RATIO + 1) <= MAX_PE_RATIO) &
                    (ranked['ROE %'].fillna(MIN_ROE - 1) >= MIN_ROE)
                ].copy()

                if not ranked_filtered.empty:
                    risk_per_trade = results_data["capital"] * (results_data["risk_pct"] / 100)
                    ranked_filtered = ranked_filtered[ranked_filtered['Entry'] != ranked_filtered['Stop-Loss']]
                    if not ranked_filtered.empty:
                        ranked_filtered['Position Size'] = np.floor(risk_per_trade / (ranked_filtered['Entry'] - ranked_filtered['Stop-Loss']))
                        ranked_filtered['Investment'] = ranked_filtered['Position Size'] * ranked_filtered['Entry']
                        st.dataframe(style_dataframe(ranked_filtered[['Stock Name', 'Price', 'Strategy', 'Reason', 'RSI', 'P/E', 'ROE %', 'Position Size', 'Investment']]).format(precision=2), use_container_width=True, hide_index=True)
                    else:
                        st.info("Scan complete, but no valid trade setups found after filtering.")
                else:
                    st.info("Scan complete, but no stocks matched the fundamental filter criteria (P/E, ROE, Price).")
            else:
                st.info("Scan complete. No stocks matched the technical criteria.")
        else:
            st.info("⬅️ Adjust parameters and click 'Run'.")

with tab2:
    st.subheader("🔬 On-Demand Stock Analysis")
    db = st.session_state.nse_stocks
    if db:
        query = st.text_input("Enter NSE symbol:", placeholder="e.g., RELIANCE, SBIN", key="single_stock_query").upper()
        if query:
            matches = [s for s in db if query in s]
            sel = None
            if not matches:
                st.error(f"No match for '{query}'.")
            elif len(matches) == 1:
                sel = matches[0]
            else:
                sel = st.radio("Multiple matches found:", matches, horizontal=True, key="single_stock_radio")

            if sel:
                with st.spinner(f"Analyzing {sel}..."):
                    data = fetch_market_data([sel])
                if data is None or data.empty:
                    st.error(f"Could not fetch data for {sel}.")
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
                            st.write("**Recommendation Stamp**")
                            news = get_stock_briefing(sel)
                            with st.spinner("Analyzing..."):
                                st.info(get_recommendation_stamp(sel, plan, news))
                            st.write("**Recent News**")
                            if news and news.get('news'):
                                for item in news['news'][:5]:
                                    st.markdown(f"- [{item['title']}]({item['url']})")
                            else:
                                st.info("No recent news found.")
                    else:
                        st.warning(f"Could not generate a plan for {sel}. (Insufficient historical data)")
    else:
        st.warning("Stock database could not be loaded. Search is disabled.")


with tab3:
    st.subheader("📋 My Live Portfolio Tracker")
    stock_db = st.session_state.nse_stocks

    with st.container(border=True):
        st.write("**Add or Update a Stock in Your Portfolio**")
        st.markdown("_To add more shares to an existing stock, simply select it from the list and enter the new transaction details._")
        with st.form("add_update_stock_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            symbol = c1.selectbox("Select Stock", options=stock_db, index=None, placeholder="Type to search...")
            new_quantity = c2.number_input("Quantity to Add", min_value=1, step=1)
            buy_price_str = c3.text_input("Buy Price (for this transaction)", placeholder="e.g., 178.50")

            submitted = st.form_submit_button("Add / Update Stock")

            if submitted and symbol and new_quantity > 0 and buy_price_str:
                try:
                    buy_price = float(buy_price_str)

                    existing_stock_index = -1
                    for i, stock in enumerate(st.session_state.portfolio):
                        if stock['symbol'] == symbol:
                            existing_stock_index = i
                            break

                    if existing_stock_index != -1:
                        existing_stock = st.session_state.portfolio[existing_stock_index]
                        old_total_value = existing_stock['quantity'] * existing_stock['buy_price']
                        old_quantity = existing_stock['quantity']
                        new_transaction_value = new_quantity * buy_price
                        new_total_quantity = old_quantity + new_quantity
                        new_total_value = old_total_value + new_transaction_value
                        new_avg_price = new_total_value / new_total_quantity

                        st.session_state.portfolio[existing_stock_index]['quantity'] = new_total_quantity
                        st.session_state.portfolio[existing_stock_index]['buy_price'] = new_avg_price
                        st.success(f"Updated {symbol}. New Qty: {new_total_quantity}, New Avg. Price: {new_avg_price:.2f}")
                    else:
                        st.session_state.portfolio.append({"symbol": symbol, "quantity": new_quantity, "buy_price": buy_price})
                        st.success(f"Added {symbol} to your portfolio.")

                    save_portfolio_to_disk(st.session_state.portfolio)
                    if 'processed_portfolio' in st.session_state:
                        del st.session_state['processed_portfolio']
                    st.rerun()

                except ValueError:
                    st.error("Invalid Buy Price. Please enter a valid number.")

    st.divider()

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add a stock to begin tracking.", icon="📈")
    else:
        if 'processed_portfolio' not in st.session_state:
            st.session_state.refresh_portfolio = True

        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.session_state.refresh_portfolio = True
            if 'portfolio_conclusion' in st.session_state:
                del st.session_state['portfolio_conclusion']

        if st.session_state.get('refresh_portfolio', False):
             with st.spinner("Fetching latest data..."):
                portfolio_symbols = [s['symbol'] for s in st.session_state.portfolio]
                portfolio_data = fetch_market_data(portfolio_symbols, period="2y")
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

            st.divider()
            st.subheader("🔍 Analysis Zone")
            selected_portfolio_stock = st.selectbox("Select a portfolio stock for detailed analysis:", options=df_portfolio['Stock Name'].tolist(), index=None, key="portfolio_stock_selector")

            if selected_portfolio_stock:
                stock_info = df_portfolio[df_portfolio['Stock Name'] == selected_portfolio_stock].to_dict('records')[0]
                chart_col, details_col = st.columns(2)
                with chart_col:
                    st.plotly_chart(create_stock_chart(stock_info['Chart Data'], stock_info['Stock Name']),use_container_width=True)
                with details_col:
                    st.write("**Recommendation Stamp**")
                    news_data = get_stock_briefing(stock_info['Stock Name'])
                    with st.spinner("Analyzing..."):
                        stamp = get_recommendation_stamp(selected_portfolio_stock, stock_info, news_data)
                        st.info(stamp)

                    st.write("**Recent News**")
                    if news_data and news_data.get('news'):
                        for item in news_data['news'][:5]: st.markdown(f"- [{item['title']}]({item['url']})")
                    else: st.info("No recent news found for this stock.")

            st.divider()
            st.subheader("🤖 AI-Powered Portfolio Conclusion")

            if 'portfolio_conclusion' in st.session_state:
                 st.markdown(st.session_state.portfolio_conclusion)
                 if st.button("Regenerate Analysis"):
                     del st.session_state['portfolio_conclusion']
                     st.rerun()
            else:
                if st.button("🚀 Generate Full Portfolio Analysis", use_container_width=True):
                    with st.spinner("Performing deep portfolio analysis... This may take a moment."):
                        conclusion = generate_portfolio_conclusion(df_portfolio)
                        st.session_state.portfolio_conclusion = conclusion
                        st.rerun()

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

with tab5:
    st.subheader("🚀 My SIP Dashboard")
    st.markdown("This dashboard tracks your real SIP journey. **Start by feeding your initial holdings.** The app will then auto-simulate all future investments.")

    if 'edit_sip_mode' not in st.session_state:
        st.session_state.edit_sip_mode = False

    if not st.session_state.sip_portfolio_state.get('holdings') or st.session_state.edit_sip_mode:
        st.warning("Please add or update your initial SIP holdings from September 2025.", icon="❗")

        with st.form("initial_sip_holdings_form"):
            st.markdown("**Instructions:** Go to your Zerodha **Kite -> Holdings** and **Coin -> Holdings** to get the exact `Qty.` and `Avg.` price for every instrument you own.")

            initial_holdings_input = []
            current_holdings_list = st.session_state.sip_portfolio_state.get('holdings', [])

            def get_current_holding(name):
                if isinstance(current_holdings_list, list):
                    for item in current_holdings_list:
                        if isinstance(item, dict) and item.get('name') == name:
                            return item
                return {}


            st.subheader("Part 1: Monthly Mutual Funds")
            for item in MONTHLY_MUTUAL_FUNDS:
                st.write(f"**{item['name']}**")
                c1, c2 = st.columns(2)
                current_holding = get_current_holding(item['name'])
                qty = c1.number_input(f"Quantity", value=current_holding.get('qty', 0.0), min_value=0.0, step=0.0001, format="%.4f", key=f"{item['name']}_q")
                avg_str = c2.text_input(f"Avg. Price", value=f"{current_holding.get('avg_price', 0.0):.4f}", key=f"{item['name']}_p")
                if qty > 0:
                    initial_holdings_input.append({"qty": qty, "avg_price_str": avg_str, "name": item['name'], "type": item['type'], "symbol": item['symbol']})

            st.subheader("Part 2: Monthly Stocks")
            for item in MONTHLY_STOCKS:
                st.write(f"**{item['name']} ({item['symbol'].replace('.NS','')})**")
                c1, c2 = st.columns(2)
                current_holding = get_current_holding(item['name'])
                qty = c1.number_input(f"Quantity", value=int(current_holding.get('qty', 0)), min_value=0, step=1, key=f"{item['name']}_q")
                avg_str = c2.text_input(f"Avg. Price", value=f"{current_holding.get('avg_price', 0.0):.2f}", key=f"{item['name']}_p")
                if qty > 0:
                    initial_holdings_input.append({"qty": qty, "avg_price_str": avg_str, "name": item['name'], "type": item['type'], "symbol": item['symbol']})
            
            st.subheader("Part 3: Monthly Gold & Silver ETFs")
            for item in MONTHLY_ETFS:
                st.write(f"**{item['name']} ({item['symbol'].replace('.NS','')})**")
                c1, c2 = st.columns(2)
                current_holding = get_current_holding(item['name'])
                qty = c1.number_input(f"Quantity", value=int(current_holding.get('qty', 0)), min_value=0, step=1, key=f"{item['name']}_q")
                avg_str = c2.text_input(f"Avg. Price", value=f"{current_holding.get('avg_price', 0.0):.2f}", key=f"{item['name']}_p")
                if qty > 0:
                    initial_holdings_input.append({"qty": qty, "avg_price_str": avg_str, "name": item['name'], "type": item['type'], "symbol": item['symbol']})

            st.subheader("Part 4: Weekly Gold & Silver ETFs")
            for item in WEEKLY_ETFS:
                st.write(f"**{item['name']} ({item['symbol'].replace('.NS','')})**")
                c1, c2 = st.columns(2)
                current_holding = get_current_holding(item['name'])
                qty = c1.number_input(f"Quantity", value=int(current_holding.get('qty', 0)), min_value=0, step=1, key=f"{item['name']}_q")
                avg_str = c2.text_input(f"Avg. Price", value=f"{current_holding.get('avg_price', 0.0):.2f}", key=f"{item['name']}_p")
                if qty > 0:
                    initial_holdings_input.append({"qty": qty, "avg_price_str": avg_str, "name": item['name'], "type": item['type'], "symbol": item['symbol']})


            submitted = st.form_submit_button("Save Holdings")

            if submitted:
                try:
                    final_holdings = []
                    for data in initial_holdings_input:
                        data['avg_price'] = float(data['avg_price_str'])
                        del data['avg_price_str'] # clean up
                        final_holdings.append(data)

                    initial_state = {
                        "last_monthly_sip_date": "2025-09-15",
                        "last_weekly_sip_date": "2025-09-15",
                        "holdings": final_holdings
                    }
                    save_sip_portfolio(initial_state)
                    st.session_state.sip_portfolio_state = initial_state
                    st.session_state.edit_sip_mode = False
                    st.success("Your holdings have been saved!")
                    st.rerun()
                except ValueError:
                    st.error("Invalid price entered. Please make sure all average prices are valid numbers (e.g., 1182.20).")

    else:
        # --- This is the main dashboard view ---
        if st.button("✏️ Edit Holdings"):
            st.session_state.edit_sip_mode = True
            st.rerun()

        with st.spinner("Fetching live market prices for your SIP holdings..."):
            holdings_data = st.session_state.sip_portfolio_state.get('holdings', [])

            if isinstance(holdings_data, dict):
                holdings_list_raw = list(holdings_data.values())
            else:
                holdings_list_raw = holdings_data

            holdings_list = [item for item in holdings_list_raw if isinstance(item, dict) and 'symbol' in item]

            symbols = [item['symbol'] for item in holdings_list]
            if symbols:
                live_data = yf.download(symbols, period="1d", progress=False)

                if live_data.empty:
                    st.warning("Could not fetch live price data for SIP holdings.", icon="⚠️")
                else:
                    if len(symbols) == 1:
                        live_prices = pd.Series({symbols[0]: live_data['Close'].iloc[-1]})
                    else:
                        live_prices = live_data['Close'].iloc[-1]

                    display_list = []
                    for item in holdings_list:
                        current_price = live_prices.get(item['symbol'], item.get('avg_price', 0))
                        qty = item.get('qty', 0)
                        avg_price = item.get('avg_price', 0)

                        buy_value = qty * avg_price
                        current_value = qty * current_price
                        pnl = current_value - buy_value
                        pnl_pct = (pnl / buy_value) * 100 if buy_value > 0 else 0

                        display_list.append({
                            "Name": item.get('name', 'N/A'), "Type": item.get('type', 'N/A'), "Qty": qty,
                            "Avg. Buy Price": avg_price, "Invested": buy_value, "CMP": current_price,
                            "Current Value": current_value, "P&L": pnl, "P&L %": pnl_pct
                        })
                    df_sip = pd.DataFrame(display_list)

                    if not df_sip.empty:
                        total_investment = df_sip['Invested'].sum()
                        total_current_value = df_sip['Current Value'].sum()
                        total_pnl = df_sip['P&L'].sum()
                        total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

                        st.subheader("📊 SIP Performance Summary")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Amount Invested", f"₹{total_investment:,.2f}")
                        m2.metric("Current Portfolio Value", f"₹{total_current_value:,.2f}")
                        m3.metric("Overall P&L", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")

                        st.divider()
                        st.subheader("📋 Detailed Holdings")
                        
                        df_monthly_mf = df_sip[df_sip['Name'].isin([item['name'] for item in MONTHLY_MUTUAL_FUNDS])]
                        df_monthly_stocks = df_sip[df_sip['Name'].isin([item['name'] for item in MONTHLY_STOCKS])]
                        df_monthly_etfs = df_sip[df_sip['Name'].isin([item['name'] for item in MONTHLY_ETFS])]
                        df_weekly_etf = df_sip[df_sip['Name'].isin([item['name'] for item in WEEKLY_ETFS])]

                        display_cols = ["Name", "Type", "Qty", "Avg. Buy Price", "CMP", "Invested", "Current Value", "P&L", "P&L %"]

                        if not df_monthly_mf.empty:
                            subtotal_mf = df_monthly_mf['Invested'].sum()
                            st.write(f"**Part 1: Monthly Mutual Funds** (Invested: ₹{subtotal_mf:,.2f})")
                            st.dataframe(style_dataframe(df_monthly_mf[display_cols]).format(precision=2), use_container_width=True, hide_index=True)
                        
                        if not df_monthly_stocks.empty:
                            subtotal_s = df_monthly_stocks['Invested'].sum()
                            st.write(f"**Part 2: Monthly Stocks** (Invested: ₹{subtotal_s:,.2f})")
                            st.dataframe(style_dataframe(df_monthly_stocks[display_cols]).format(precision=2), use_container_width=True, hide_index=True)

                        if not df_monthly_etfs.empty:
                            subtotal_me = df_monthly_etfs['Invested'].sum()
                            st.write(f"**Part 3: Monthly Gold & Silver ETFs** (Invested: ₹{subtotal_me:,.2f})")
                            st.dataframe(style_dataframe(df_monthly_etfs[display_cols]).format(precision=2), use_container_width=True, hide_index=True)

                        if not df_weekly_etf.empty:
                            subtotal_we = df_weekly_etf['Invested'].sum()
                            st.write(f"**Part 4: Weekly Gold & Silver ETFs** (Invested: ₹{subtotal_we:,.2f})")
                            st.dataframe(style_dataframe(df_weekly_etf[display_cols]).format(precision=2), use_container_width=True, hide_index=True)
            else:
                st.info("Your SIP portfolio is configured but currently has no holdings with a quantity greater than zero.")

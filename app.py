import os
import json
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random
from functools import lru_cache

# ==================== ALPHA VANTAGE SETUP ====================
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# 1. SETUP SESSION (Prevents Rate Limits)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

app = Flask(__name__)
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*", "https://aurum-au.com"])

# 2. INITIALIZATION
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_api_symbol(symbol):
    """Convert symbol to API-specific format"""
    if symbol in ['BTC', 'ETH', 'SOL', 'BNB'] and not symbol.endswith('-USD'):
        return f"{symbol}-USD"
    if any(s in symbol for s in ['BBCA', 'TLKM', 'ASII']) and not symbol.endswith('.JK'):
        return f"{symbol}.JK"
    return symbol

def get_alpha_vantage_symbol(symbol):
    """Convert symbol to Alpha Vantage format"""
    # Alpha Vantage uses different formats for crypto
    crypto_map = {
        'BTC': 'BTC',
        'ETH': 'ETH',
        'SOL': 'SOL',
        'BNB': 'BNB'
    }
    
    if symbol in crypto_map:
        return crypto_map[symbol]
    
    # For stocks, add exchange suffix if needed
    if symbol in ['BBCA', 'TLKM', 'ASII']:
        return f"{symbol}.JK"
    
    return symbol

# ==================== ALPHA VANTAGE FUNCTIONS ====================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def get_alpha_vantage_price(symbol):
    """Get current price from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    av_symbol = get_alpha_vantage_symbol(symbol)
    
    try:
        # For crypto
        if symbol in ['BTC', 'ETH', 'SOL', 'BNB']:
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={av_symbol}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
            response = session.get(url, timeout=10)
            data = response.json()
            
            if "Realtime Currency Exchange Rate" in data:
                return float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        
        # For stocks
        else:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = session.get(url, timeout=10)
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                return float(data["Global Quote"]["05. price"])
    
    except Exception as e:
        print(f"Alpha Vantage error for {symbol}: {str(e)}")
    
    return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def get_alpha_vantage_historical(symbol, start_date="2013-01-01"):
    """Get historical data from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    av_symbol = get_alpha_vantage_symbol(symbol)
    
    try:
        # Determine function based on symbol type
        if symbol in ['BTC', 'ETH', 'SOL', 'BNB']:
            function = "DIGITAL_CURRENCY_DAILY"
            market = "USD"
            url = f"https://www.alphavantage.co/query?function={function}&symbol={av_symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}"
        else:
            function = "TIME_SERIES_DAILY"
            url = f"https://www.alphavantage.co/query?function={function}&symbol={av_symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        
        response = session.get(url, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            print(f"Alpha Vantage error: {data['Error Message']}")
            return None
        
        # Parse response based on function
        if symbol in ['BTC', 'ETH', 'SOL', 'BNB']:
            time_series = data.get("Time Series (Digital Currency Daily)", {})
        else:
            time_series = data.get("Time Series (Daily)", {})
        
        # Convert to list of dicts
        historical_data = []
        for date_str, values in time_series.items():
            if date_str < start_date:
                continue
            
            try:
                if symbol in ['BTC', 'ETH', 'SOL', 'BNB']:
                    close_price = float(values.get(f"4a. close (USD)", 0))
                else:
                    close_price = float(values.get("4. close", 0))
                
                historical_data.append({
                    "date": date_str,
                    "close": round(close_price, 2)
                })
            except:
                continue
        
        # Sort by date
        historical_data.sort(key=lambda x: x['date'])
        return historical_data
    
    except Exception as e:
        print(f"Alpha Vantage historical error for {symbol}: {str(e)}")
    
    return None

# ==================== HYBRID PRICE FUNCTIONS ====================

def get_price_safe(symbol):
    """Get price from multiple sources with fallback"""
    sources = [
        get_cached_price,          # Firebase cache first
        get_alpha_vantage_price,   # Alpha Vantage second
        get_yfinance_price_fallback # yFinance as last resort
    ]
    
    for source in sources:
        try:
            price = source(symbol)
            if price and price > 0:
                return price
        except Exception as e:
            print(f"Price source failed ({source.__name__}): {str(e)}")
            continue
    
    return 0

def get_yfinance_price_fallback(symbol):
    """Fallback to yfinance"""
    try:
        ticker = yf.Ticker(get_api_symbol(symbol), session=session)
        price = ticker.fast_info.get('last_price', 0)
        return price if price and price > 0 else None
    except:
        return None

@lru_cache(maxsize=100)
def get_cached_price(symbol, expiry_minutes=5):
    """Cache price data with multi-source support"""
    cache_key = f"price_{symbol}"
    cache_ref = db.collection('cache').document(cache_key)
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        cache_data = cache_doc.to_dict()
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cached_time < timedelta(minutes=expiry_minutes):
            return cache_data['price']
    
    # Get fresh price from best available source
    price = None
    
    # Try Alpha Vantage first (more reliable)
    if ALPHA_VANTAGE_API_KEY:
        price = get_alpha_vantage_price(symbol)
    
    # Fallback to yfinance
    if not price or price == 0:
        price = get_yfinance_price_fallback(symbol)
    
    if price and price > 0:
        cache_ref.set({
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'source': 'alpha_vantage' if ALPHA_VANTAGE_API_KEY else 'yfinance'
        })
    
    return price or 0

# ==================== HELPER FUNCTIONS ====================

def df_to_list(df):
    if df.empty:
        return []
    
    data_list = []
    for date, row in df.iterrows():
        try:
            # Try different column names
            if 'Close' in row:
                close_val = row['Close']
            elif 'close' in row:
                close_val = row['close']
            elif 'Adj Close' in row:
                close_val = row['Adj Close']
            else:
                # Get the last numeric column
                numeric_cols = [col for col in row.index if pd.api.types.is_numeric_dtype(type(row[col]))]
                close_val = row[numeric_cols[-1]] if numeric_cols else None
            
            if close_val is not None:
                # Handle Series vs scalar
                if hasattr(close_val, 'iloc'):
                    close_val = close_val.iloc[0]
                
                data_list.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "close": round(float(close_val), 2)
                })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return data_list

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception, ConnectionError, TimeoutError)),
    before_sleep=lambda retry_state: print(f"Retrying yfinance request... Attempt {retry_state.attempt_number}")
)
def safe_yf_download(symbol, start, interval, session):
    """Safe download with retry logic"""
    time.sleep(random.uniform(1, 3))
    return yf.download(symbol, start=start, interval=interval, session=session)

# ==================== ENDPOINTS ====================

@app.route('/get-price', methods=['GET'])
def get_price():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        price = get_price_safe(symbol)
        return jsonify({
            "symbol": symbol,
            "price": price,
            "source": "multi-source"  # Indicates hybrid approach
        })
    except Exception as e:
        print(f"Error fetching price for {symbol}: {str(e)}")
        return jsonify({"error": str(e), "symbol": symbol}), 500

@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        # Rate limiting check
        rate_limit_key = f"rate_limit_{symbol}"
        rate_ref = db.collection('rate_limits').document(rate_limit_key)
        rate_doc = rate_ref.get()
        
        if rate_doc.exists:
            last_call = datetime.fromisoformat(rate_doc.to_dict()['last_call'])
            if datetime.now() - last_call < timedelta(minutes=5):
                return jsonify({
                    "status": "rate_limited", 
                    "message": "Please wait 5 minutes before syncing this symbol again"
                }), 429
        
        # Update rate limit timestamp
        rate_ref.set({'last_call': datetime.now().isoformat()})
        
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()

        start_sync = "2013-01-01"
        existing_daily = doc.to_dict().get('daily', []) if doc.exists else []
        existing_monthly = doc.to_dict().get('monthly', []) if doc.exists else []
        
        if existing_daily:
            start_sync = (datetime.strptime(existing_daily[-1]['date'], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Check if already up to date (within 1 day)
        if existing_daily:
            last_date = datetime.strptime(existing_daily[-1]['date'], '%Y-%m-%d')
            if datetime.now().date() - last_date.date() <= timedelta(days=1):
                return jsonify({
                    "status": "already_updated", 
                    "message": "Data is already up to date",
                    "count": len(existing_daily)
                })

        try:
            # First try Alpha Vantage for historical data if available
            new_data_daily = None
            if ALPHA_VANTAGE_API_KEY:
                print(f"Trying Alpha Vantage for {symbol}...")
                new_data_daily = get_alpha_vantage_historical(symbol, start_sync)
            
            # If Alpha Vantage fails or not available, use yfinance
            if not new_data_daily:
                print(f"Using yfinance for {symbol}...")
                new_d = safe_yf_download(api_symbol, start=start_sync, interval="1d", session=session)
                time.sleep(random.uniform(2, 4))
                new_m = safe_yf_download(api_symbol, start=start_sync, interval="1mo", session=session)
                
                new_data_daily = df_to_list(new_d)
                new_data_monthly = df_to_list(new_m)
            else:
                # For Alpha Vantage, we only get daily data
                # Create monthly data by sampling (every 30 days)
                new_data_monthly = []
                for i, item in enumerate(new_data_daily):
                    if i % 30 == 0:
                        new_data_monthly.append(item)
            
            updated_daily = existing_daily + new_data_daily
            updated_monthly = existing_monthly + new_data_monthly

            # Deduplicate by date
            daily_dict = {item['date']: item for item in updated_daily}
            monthly_dict = {item['date']: item for item in updated_monthly}
            
            updated_daily = list(daily_dict.values())
            updated_monthly = list(monthly_dict.values())
            
            # Sort by date
            updated_daily.sort(key=lambda x: x['date'])
            updated_monthly.sort(key=lambda x: x['date'])

            asset_ref.set({
                "daily": updated_daily[-1000:],
                "monthly": updated_monthly[-120:],
                "last_updated": datetime.now().isoformat(),
                "data_source": "alpha_vantage" if ALPHA_VANTAGE_API_KEY and new_data_daily else "yfinance"
            }, merge=True)
            
            return jsonify({
                "status": "synced", 
                "count": len(updated_daily),
                "new_points": len(updated_daily) - len(existing_daily),
                "source": "alpha_vantage" if ALPHA_VANTAGE_API_KEY and new_data_daily else "yfinance"
            })
            
        except Exception as e:  # Changed from YahooFinanceException
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                return jsonify({
                    "error": "Data source rate limit exceeded",
                    "message": "Please try again in 30 minutes",
                    "retry_after": 1800
                }), 429
            else:
                raise e
            
    except Exception as e:
        print(f"Error in get-historical-data for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval') or request.args.get('timeframe', 'daily')
    api_symbol = get_api_symbol(symbol)

    try:
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        if not asset_doc.exists: 
            return jsonify({"error": "No data found. Sync first."}), 404
        
        full_data = asset_doc.to_dict().get(interval, [])
        df = pd.DataFrame(full_data)
        
        # Sampling Logic
        if interval == 'daily':
            df_window = df.tail(126).copy() # 6 months
            df_sampled = df_window.iloc[::4].tail(30) 
        else:
            df_window = df.tail(48).copy() # 4 years
            df_sampled = df_window

        # Technical Indicators
        df_window['MA20'] = df_window['close'].rolling(window=20).mean()
        curr_ma = round(float(df_window['MA20'].iloc[-1]), 2)
        
        # Fetch News (with session) - only if yfinance available
        news = []
        try:
            news = [n['title'] for n in yf.Ticker(api_symbol, session=session).news[:5]]
        except:
            news = ["News data temporarily unavailable"]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior analyst. Respond ONLY in valid JSON. Fields: trend, patterns (array), sentiment_score (0-100), verdict, suggestion (BUY/HOLD/SELL)."},
                {"role": "user", "content": f"Analyze {api_symbol} {interval} data.\nCSV Data:\n{df_sampled.to_csv()}\nIndicators: MA20 is {curr_ma}.\nNews: {news}"}
            ],
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "alpha_vantage_available": bool(ALPHA_VANTAGE_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
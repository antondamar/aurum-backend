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
import random
import numpy as np

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
    """Convert symbol to yfinance format"""
    if not symbol:
        return ""
    
    s = symbol.upper().strip()
    cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'LINK', 'LTC']
    if s in cryptos:
        return f"{s}-USD"
    return s

# ==================== TECHNICAL ANALYSIS FUNCTIONS ====================

def calculate_fibonacci_retracement(df):
    """Calculate Fibonacci retracement levels"""
    if len(df) < 20:
        return {
            'levels': {},
            'current_price': 0,
            'current_zone': "Insufficient data",
            'distance_from_high': 0,
            'distance_from_low': 0
        }
    
    try:
        # Use last 20 days for swing high/low
        recent_data = df.tail(20)
        swing_high = float(recent_data['close'].max())
        swing_low = float(recent_data['close'].min())
        
        # Fibonacci levels
        diff = swing_high - swing_low
        levels = {
            'swing_high': swing_high,
            'swing_low': swing_low,
            '0.236': swing_high - diff * 0.236,
            '0.382': swing_high - diff * 0.382,
            '0.5': swing_high - diff * 0.5,
            '0.618': swing_high - diff * 0.618,
            '0.786': swing_high - diff * 0.786,
            '1.0': swing_low
        }
        
        # Convert all values to float for JSON serialization
        for key in levels:
            levels[key] = float(levels[key])
        
        # Current position relative to Fibonacci levels
        current_price = float(df['close'].iloc[-1])
        
        # Find which Fibonacci zone current price is in
        fib_zones = list(levels.values())
        fib_zones.sort(reverse=True)
        
        current_zone = None
        for i in range(len(fib_zones)-1):
            if fib_zones[i] >= current_price >= fib_zones[i+1]:
                # Determine the Fibonacci level
                for key, value in levels.items():
                    if abs(value - fib_zones[i]) < 0.001:
                        if key in ['0.382', '0.5', '0.618']:
                            current_zone = f"Fibonacci {key} retracement zone"
                        else:
                            current_zone = f"Near {key} level"
                        break
                break
        
        return {
            'levels': levels,
            'current_price': current_price,
            'current_zone': current_zone or "Outside main Fibonacci zones",
            'distance_from_high': round(((swing_high - current_price) / diff) * 100, 2) if diff > 0 else 0,
            'distance_from_low': round(((current_price - swing_low) / diff) * 100, 2) if diff > 0 else 0
        }
    except Exception as e:
        print(f"Fibonacci calculation error: {e}")
        return {
            'levels': {},
            'current_price': 0,
            'current_zone': "Calculation error",
            'distance_from_high': 0,
            'distance_from_low': 0
        }

def calculate_moving_averages(df):
    """Calculate various moving averages"""
    ma_data = {}
    
    try:
        # Ensure we have enough data
        if len(df) >= 50:
            df = df.copy()
            ma20 = df['close'].rolling(window=20).mean().iloc[-1]
            ma50 = df['close'].rolling(window=50).mean().iloc[-1]
            ma200 = df['close'].rolling(window=200).mean().iloc[-1]
            
            ma_data['MA20'] = float(round(ma20, 2)) if not pd.isna(ma20) else 0
            ma_data['MA50'] = float(round(ma50, 2)) if not pd.isna(ma50) else 0
            ma_data['MA200'] = float(round(ma200, 2)) if not pd.isna(ma200) else 0
            
            # Determine MA trend
            current_price = float(df['close'].iloc[-1])
            if ma_data['MA20'] > 0 and ma_data['MA50'] > 0:
                ma_data['ma_trend'] = "Bullish" if current_price > ma_data['MA20'] > ma_data['MA50'] else "Bearish"
            else:
                ma_data['ma_trend'] = "Unknown"
            
            # Convert boolean to string for JSON serialization
            ma_data['golden_cross'] = "Yes" if ma_data.get('MA50', 0) > ma_data.get('MA200', 0) else "No"
            ma_data['price_vs_ma20'] = "Above" if current_price > ma_data.get('MA20', 0) else "Below"
        else:
            ma_data['MA20'] = 0
            ma_data['MA50'] = 0
            ma_data['MA200'] = 0
            ma_data['ma_trend'] = "Insufficient data"
            ma_data['golden_cross'] = "Unknown"
            ma_data['price_vs_ma20'] = "Unknown"
    except Exception as e:
        print(f"Moving average calculation error: {e}")
        ma_data = {
            'MA20': 0,
            'MA50': 0,
            'MA200': 0,
            'ma_trend': "Calculation error",
            'golden_cross': "Unknown",
            'price_vs_ma20': "Unknown"
        }
    
    return ma_data

def analyze_support_resistance(df):
    """Identify key support and resistance levels"""
    if len(df) < 30:
        return {
            'support': 0,
            'resistance': 0,
            'distance_to_resistance_pct': 0,
            'distance_to_support_pct': 0,
            'closest_level': "Insufficient data"
        }
    
    try:
        recent = df.tail(30)
        
        # Simple support/resistance detection
        closes = recent['close'].values
        resistance = float(np.max(closes[-10:]))  # Recent high
        support = float(np.min(closes[-10:]))     # Recent low
        
        current_price = float(closes[-1])
        distance_to_resistance = round(((resistance - current_price) / current_price) * 100, 2) if current_price > 0 else 0
        distance_to_support = round(((current_price - support) / current_price) * 100, 2) if current_price > 0 else 0
        
        return {
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'distance_to_resistance_pct': distance_to_resistance,
            'distance_to_support_pct': distance_to_support,
            'closest_level': "Resistance" if distance_to_resistance < distance_to_support else "Support"
        }
    except Exception as e:
        print(f"Support/resistance calculation error: {e}")
        return {
            'support': 0,
            'resistance': 0,
            'distance_to_resistance_pct': 0,
            'distance_to_support_pct': 0,
            'closest_level': "Calculation error"
        }

# ==================== ENDPOINTS ====================

@app.route('/get-price', methods=['GET'])
def get_price():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        ticker = yf.Ticker(get_api_symbol(symbol), session=session)
        price = ticker.fast_info.get('last_price', 0)
        return jsonify({
            "symbol": symbol,
            "price": float(round(price, 2)),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        api_symbol = get_api_symbol(symbol)
        if not api_symbol:
            return jsonify({"error": "Invalid symbol"}), 400
        
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()
        
        existing_daily = []
        if doc.exists:
            data_dict = doc.to_dict()
            existing_daily = data_dict.get('daily', []) if data_dict else []
        
        # Check if already up to date (within 1 day)
        if existing_daily:
            try:
                last_date = datetime.strptime(existing_daily[-1]['date'], '%Y-%m-%d')
                if datetime.now().date() - last_date.date() <= timedelta(days=1):
                    return jsonify({
                        "status": "already_updated", 
                        "message": "Data is already up to date",
                        "count": len(existing_daily)
                    })
            except:
                pass
        
        # Fetch new data
        ticker = yf.Ticker(api_symbol, session=session)
        hist = ticker.history(period="1mo", interval="1d")
        
        new_data = []
        for date, row in hist.iterrows():
            new_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "close": float(round(row['Close'], 2))
            })
        
        # Update Firebase
        updated_daily = existing_daily + new_data
        # Deduplicate
        daily_dict = {}
        for item in updated_daily:
            if 'date' in item:
                daily_dict[item['date']] = item
        
        updated_daily = list(daily_dict.values())
        updated_daily.sort(key=lambda x: x['date'])
        
        # Truncate to last 1000 entries
        updated_daily = updated_daily[-1000:]
        
        # Save to Firebase
        asset_ref.set({
            "daily": updated_daily,
            "last_updated": datetime.now().isoformat()
        }, merge=True)
        
        return jsonify({
            "status": "synced", 
            "count": len(updated_daily),
            "new_points": len(new_data)
        })
        
    except Exception as e:
        print(f"Error in get-historical-data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    interval = request.args.get('interval', 'daily')
    
    # Force interval to daily/monthly
    if 'monthly' in interval.lower() or '1mo' in interval.lower():
        interval_type = 'monthly'
        window_size = 48  # 4 years
    else:
        interval_type = 'daily'
        window_size = 126  # 6 months
    
    api_symbol = get_api_symbol(symbol)
    
    try:
        # Get data from Firebase
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        if not asset_doc.exists:
            return jsonify({"error": "No data found. Sync first."}), 404
        
        data_dict = asset_doc.to_dict()
        if not data_dict:
            return jsonify({"error": "No data found. Sync first."}), 404
        
        full_data = data_dict.get('daily', [])
        if not full_data or len(full_data) < window_size:
            return jsonify({"error": f"Insufficient data. Need at least {window_size} days."}), 404
        
        # Convert to DataFrame
        df = pd.DataFrame(full_data)
        
        # Ensure we have enough data
        if len(df) < window_size:
            return jsonify({"error": "Not enough historical data"}), 404
        
        # Get analysis window
        df_window = df.tail(window_size).copy()
        
        # Ensure close column exists and is numeric
        if 'close' not in df_window.columns:
            return jsonify({"error": "Invalid data format: missing 'close' column"}), 500
        
        df_window['close'] = pd.to_numeric(df_window['close'], errors='coerce')
        df_window = df_window.dropna(subset=['close'])
        
        # Calculate technical indicators
        ma_data = calculate_moving_averages(df_window)
        fib_data = calculate_fibonacci_retracement(df_window)
        sr_data = analyze_support_resistance(df_window)
        
        # Get recent news
        news_titles = []
        news_sentiment = "neutral"
        try:
            ticker = yf.Ticker(api_symbol, session=session)
            news = ticker.news[:10] if hasattr(ticker, 'news') else []
            news_titles = [str(n.get('title', '')) for n in news if 'title' in n]
            
            # Simple sentiment analysis
            positive_words = ['bullish', 'gain', 'rise', 'surge', 'rally', 'positive', 'strong', 'up', 'increase']
            negative_words = ['bearish', 'drop', 'fall', 'plunge', 'decline', 'negative', 'weak', 'down', 'decrease']
            
            positive_count = sum(1 for title in news_titles 
                                for word in positive_words if word.lower() in str(title).lower())
            negative_count = sum(1 for title in news_titles 
                                for word in negative_words if word.lower() in str(title).lower())
            
            if positive_count > negative_count:
                news_sentiment = "positive"
            elif negative_count > positive_count:
                news_sentiment = "negative"
        except Exception as e:
            print(f"News fetching error: {e}")
            news_titles = ["No recent news available"]
        
        # Prepare data for AI - ensure all values are JSON serializable
        current_price = float(df_window['close'].iloc[-1]) if len(df_window) > 0 else 0
        
        analysis_data = {
            "symbol": str(symbol),
            "current_price": current_price,
            "moving_averages": ma_data,
            "fibonacci": fib_data,
            "support_resistance": sr_data,
            "news_sentiment": str(news_sentiment),
            "recent_news": [str(news) for news in news_titles[:5]],
            "analysis_date": datetime.now().isoformat()
        }
        
        # Create prompt for AI
        prompt = f"""
        Analyze {symbol} ({interval_type} data) with the following information:
        
        Current Price: ${analysis_data['current_price']}
        
        MOVING AVERAGES:
        - MA20: ${ma_data.get('MA20', 'N/A')}
        - MA50: ${ma_data.get('MA50', 'N/A')}
        - MA200: ${ma_data.get('MA200', 'N/A')}
        - Price is currently {ma_data.get('price_vs_ma20', 'N/A')} the 20-day MA
        - Trend: {ma_data.get('ma_trend', 'N/A')}
        
        FIBONACCI RETRACEMENT (based on last 20 days):
        - Swing High: ${fib_data.get('levels', {}).get('swing_high', 'N/A')}
        - Swing Low: ${fib_data.get('levels', {}).get('swing_low', 'N/A')}
        - Current Zone: {fib_data.get('current_zone', 'N/A')}
        
        SUPPORT & RESISTANCE:
        - Support: ${sr_data.get('support', 'N/A')}
        - Resistance: ${sr_data.get('resistance', 'N/A')}
        - Closest to: {sr_data.get('closest_level', 'N/A')}
        
        NEWS SENTIMENT: {news_sentiment.upper()}
        
        RECENT NEWS HEADLINES:
        {chr(10).join(analysis_data['recent_news'])}
        
        Based on ALL the above technical indicators and news sentiment, provide analysis in JSON format with these exact fields:
        - trend: (string) Overall market trend
        - patterns: (array of strings) Key technical patterns
        - fibonacci_interpretation: (string) Analysis of Fibonacci levels
        - ma_analysis: (string) Moving average analysis
        - news_analysis: (string) News sentiment impact
        - risk_level: (string) Low/Medium/High
        - verdict: (string) Detailed analysis conclusion
        - suggestion: (string) BUY/HOLD/SELL
        - confidence_score: (number between 0-100)
        - recommended_action: (string) Specific action to take
        """
        
        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior technical analyst. Respond ONLY in valid JSON format with the exact fields requested. Provide specific, actionable insights based on all technical indicators combined."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        
        # Ensure all values are JSON serializable
        sanitized_ai_response = {}
        for key, value in ai_response.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized_ai_response[key] = value
            elif isinstance(value, list):
                sanitized_ai_response[key] = [str(item) for item in value]
            else:
                sanitized_ai_response[key] = str(value)
        
        # Combine AI response with technical data
        result = {
            **sanitized_ai_response,
            "technical_data": analysis_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get-ai-insight: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
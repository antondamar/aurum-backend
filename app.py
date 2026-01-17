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
    s = symbol.upper()
    cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'LINK', 'LTC']
    if s in cryptos:
        return f"{s}-USD"
    return s

# ==================== TECHNICAL ANALYSIS FUNCTIONS ====================

def calculate_fibonacci_retracement(df):
    """Calculate Fibonacci retracement levels"""
    if len(df) < 20:
        return None
    
    # Use last 20 days for swing high/low
    recent_data = df.tail(20)
    swing_high = recent_data['close'].max()
    swing_low = recent_data['close'].min()
    
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
    
    # Current position relative to Fibonacci levels
    current_price = df['close'].iloc[-1]
    
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

def calculate_moving_averages(df):
    """Calculate various moving averages"""
    ma_data = {}
    
    # Ensure we have enough data
    if len(df) >= 50:
        df = df.copy()
        ma_data['MA20'] = round(df['close'].rolling(window=20).mean().iloc[-1], 2)
        ma_data['MA50'] = round(df['close'].rolling(window=50).mean().iloc[-1], 2)
        ma_data['MA200'] = round(df['close'].rolling(window=200).mean().iloc[-1], 2)
        
        # Determine MA trend
        current_price = df['close'].iloc[-1]
        ma_data['ma_trend'] = "Bullish" if current_price > ma_data['MA20'] > ma_data['MA50'] else "Bearish"
        ma_data['golden_cross'] = ma_data['MA50'] > ma_data['MA200']
        ma_data['price_vs_ma20'] = "Above" if current_price > ma_data['MA20'] else "Below"
    
    return ma_data

def analyze_support_resistance(df):
    """Identify key support and resistance levels"""
    if len(df) < 30:
        return {}
    
    recent = df.tail(30)
    
    # Simple support/resistance detection
    closes = recent['close'].values
    resistance = np.max(closes[-10:])  # Recent high
    support = np.min(closes[-10:])     # Recent low
    
    current_price = closes[-1]
    distance_to_resistance = round(((resistance - current_price) / current_price) * 100, 2)
    distance_to_support = round(((current_price - support) / current_price) * 100, 2)
    
    return {
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'distance_to_resistance_pct': distance_to_resistance,
        'distance_to_support_pct': distance_to_support,
        'closest_level': "Resistance" if distance_to_resistance < distance_to_support else "Support"
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
            "price": round(price, 2),
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
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()
        
        existing_daily = doc.to_dict().get('daily', []) if doc.exists else []
        
        # Check if already up to date (within 1 day)
        if existing_daily:
            last_date = datetime.strptime(existing_daily[-1]['date'], '%Y-%m-%d')
            if datetime.now().date() - last_date.date() <= timedelta(days=1):
                return jsonify({
                    "status": "already_updated", 
                    "message": "Data is already up to date",
                    "count": len(existing_daily)
                })
        
        # Fetch new data
        ticker = yf.Ticker(api_symbol, session=session)
        hist = ticker.history(period="1mo", interval="1d")
        
        new_data = []
        for date, row in hist.iterrows():
            new_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "close": round(float(row['Close']), 2)
            })
        
        # Update Firebase
        updated_daily = existing_daily + new_data
        # Deduplicate
        daily_dict = {item['date']: item for item in updated_daily}
        updated_daily = list(daily_dict.values())
        updated_daily.sort(key=lambda x: x['date'])
        
        asset_ref.set({
            "daily": updated_daily[-1000:],
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
        
        full_data = asset_doc.to_dict().get('daily', [])
        if not full_data or len(full_data) < window_size:
            return jsonify({"error": f"Insufficient data. Need at least {window_size} days."}), 404
        
        df = pd.DataFrame(full_data)
        
        # Ensure we have enough data
        if len(df) < window_size:
            return jsonify({"error": "Not enough historical data"}), 404
        
        # Get analysis window
        df_window = df.tail(window_size).copy()
        
        # Calculate technical indicators
        ma_data = calculate_moving_averages(df_window)
        fib_data = calculate_fibonacci_retracement(df_window)
        sr_data = analyze_support_resistance(df_window)
        
        # Get recent news
        news_titles = []
        news_sentiment = "neutral"
        try:
            news = yf.Ticker(api_symbol, session=session).news[:10]
            news_titles = [n['title'] for n in news if 'title' in n]
            
            # Simple sentiment analysis
            positive_words = ['bullish', 'gain', 'rise', 'surge', 'rally', 'positive', 'strong']
            negative_words = ['bearish', 'drop', 'fall', 'plunge', 'decline', 'negative', 'weak']
            
            positive_count = sum(1 for title in news_titles 
                                for word in positive_words if word.lower() in title.lower())
            negative_count = sum(1 for title in news_titles 
                                for word in negative_words if word.lower() in title.lower())
            
            if positive_count > negative_count:
                news_sentiment = "positive"
            elif negative_count > positive_count:
                news_sentiment = "negative"
        except:
            news_titles = ["No recent news available"]
        
        # Prepare data for AI
        analysis_data = {
            "symbol": symbol,
            "current_price": round(df_window['close'].iloc[-1], 2),
            "moving_averages": ma_data,
            "fibonacci": fib_data,
            "support_resistance": sr_data,
            "news_sentiment": news_sentiment,
            "recent_news": news_titles[:5],
            "data_sample": df_window[['date', 'close']].tail(30).to_dict('records')
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
        - Swing High: ${fib_data['levels']['swing_high'] if fib_data else 'N/A'}
        - Swing Low: ${fib_data['levels']['swing_low'] if fib_data else 'N/A'}
        - Current Zone: {fib_data['current_zone'] if fib_data else 'N/A'}
        - Distance from High: {fib_data['distance_from_high'] if fib_data else 'N/A'}%
        - Distance from Low: {fib_data['distance_from_low'] if fib_data else 'N/A'}%
        
        SUPPORT & RESISTANCE:
        - Support: ${sr_data.get('support', 'N/A')}
        - Resistance: ${sr_data.get('resistance', 'N/A')}
        - Closest to: {sr_data.get('closest_level', 'N/A')}
        
        NEWS SENTIMENT: {news_sentiment.upper()}
        
        RECENT NEWS HEADLINES:
        {chr(10).join(news_titles[:5])}
        
        Based on ALL the above technical indicators and news sentiment, provide:
        1. Overall trend assessment
        2. Key patterns observed
        3. Fibonacci interpretation
        4. Moving average analysis
        5. News impact analysis
        6. Risk assessment
        7. Final verdict and recommendation
        
        IMPORTANT: Respond in valid JSON format with these exact fields:
        - trend: (string) Overall market trend
        - patterns: (array of strings) Key technical patterns
        - fibonacci_interpretation: (string) Analysis of Fibonacci levels
        - ma_analysis: (string) Moving average analysis
        - news_analysis: (string) News sentiment impact
        - risk_level: (string) Low/Medium/High
        - verdict: (string) Detailed analysis conclusion
        - suggestion: (string) BUY/HOLD/SELL
        - confidence_score: (number 0-100)
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
        
        # Combine AI response with technical data
        result = {
            **ai_response,
            "technical_data": {
                "current_price": analysis_data['current_price'],
                "moving_averages": ma_data,
                "fibonacci": fib_data,
                "support_resistance": sr_data,
                "news_sentiment": news_sentiment,
                "recent_news": news_titles[:5],
                "analysis_date": datetime.now().isoformat()
            }
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
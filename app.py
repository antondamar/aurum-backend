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
from typing import Dict, List, Optional, Tuple

# ==================== CONFIGURATION ====================
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# 1. SETUP SESSION
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

app = Flask(__name__)
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*", "https://aurum-au.com"])

# 2. INITIALIZATION
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_api_symbol(symbol: str) -> str:
    """Convert symbol to yfinance format"""
    if not symbol:
        return ""
    
    s = symbol.upper().strip()
    cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'LINK', 'LTC']
    if s in cryptos:
        return f"{s}-USD"
    return s

def format_currency(value: float) -> str:
    """Format currency with commas"""
    if value >= 1000:
        return f"${value:,.2f}"
    return f"${value:.2f}"

# ==================== ALPHA VANTAGE FUNCTIONS ====================

def get_alpha_vantage_news(symbol: str) -> List[Dict]:
    """Get news from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return []
    
    try:
        # Alpha Vantage news endpoint
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&limit=10"
        response = session.get(url, timeout=10)
        data = response.json()
        
        news_items = []
        if 'feed' in data:
            for item in data['feed'][:5]:  # Get top 5 news
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'time_published': item.get('time_published', ''),
                    'sentiment_score': item.get('overall_sentiment_score', 0),
                    'sentiment_label': item.get('overall_sentiment_label', 'neutral')
                })
        
        return news_items
    except Exception as e:
        print(f"Alpha Vantage news error: {e}")
        return []

# ==================== TECHNICAL ANALYSIS FUNCTIONS ====================

def calculate_moving_averages(df: pd.DataFrame) -> Dict:
    """Calculate various moving averages with proper validation"""
    ma_data = {}
    
    try:
        if len(df) >= 200:
            closes = df['close'].values
            
            # Calculate moving averages using pandas rolling
            ma20 = pd.Series(closes).rolling(window=20).mean()
            ma50 = pd.Series(closes).rolling(window=50).mean()
            ma200 = pd.Series(closes).rolling(window=200).mean()
            
            current_price = float(closes[-1])
            
            # Get last valid values
            ma20_val = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else 0
            ma50_val = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else 0
            ma200_val = float(ma200.iloc[-1]) if not pd.isna(ma200.iloc[-1]) else 0
            
            # Determine trend
            if ma20_val > 0 and ma50_val > 0 and ma200_val > 0:
                # Golden/Death cross detection
                golden_cross = ma50_val > ma200_val
                
                # Price position relative to MAs
                above_ma20 = current_price > ma20_val
                above_ma50 = current_price > ma50_val
                above_ma200 = current_price > ma200_val
                
                # Determine overall trend
                if above_ma20 and above_ma50 and above_ma200:
                    trend = "Strong Bullish"
                elif above_ma20 and above_ma50:
                    trend = "Bullish"
                elif not above_ma20 and not above_ma50:
                    trend = "Bearish"
                else:
                    trend = "Neutral"
                
                ma_data = {
                    'MA20': ma20_val,
                    'MA50': ma50_val,
                    'MA200': ma200_val,
                    'trend': trend,
                    'golden_cross': "Yes" if golden_cross else "No",
                    'price_vs_ma20': "Above" if above_ma20 else "Below",
                    'price_vs_ma50': "Above" if above_ma50 else "Below",
                    'price_vs_ma200': "Above" if above_ma200 else "Below",
                    'ma_alignment': "Bullish" if ma20_val > ma50_val > ma200_val else "Bearish" if ma20_val < ma50_val < ma200_val else "Mixed"
                }
            else:
                ma_data = {
                    'MA20': 0,
                    'MA50': 0,
                    'MA200': 0,
                    'trend': "Insufficient data",
                    'golden_cross': "Unknown",
                    'price_vs_ma20': "Unknown",
                    'price_vs_ma50': "Unknown",
                    'price_vs_ma200': "Unknown",
                    'ma_alignment': "Unknown"
                }
        else:
            ma_data = {
                'MA20': 0,
                'MA50': 0,
                'MA200': 0,
                'trend': f"Need {200-len(df)} more days for full analysis",
                'golden_cross': "Unknown",
                'price_vs_ma20': "Unknown",
                'price_vs_ma50': "Unknown",
                'price_vs_ma200': "Unknown",
                'ma_alignment': "Unknown"
            }
            
    except Exception as e:
        print(f"Moving average calculation error: {e}")
        ma_data = {
            'MA20': 0,
            'MA50': 0,
            'MA200': 0,
            'trend': "Calculation error",
            'golden_cross': "Unknown",
            'price_vs_ma20': "Unknown",
            'price_vs_ma50': "Unknown",
            'price_vs_ma200': "Unknown",
            'ma_alignment': "Unknown"
        }
    
    return ma_data

def calculate_fibonacci_retracement(df: pd.DataFrame) -> Dict:
    """Calculate Fibonacci retracement levels with zone detection"""
    if len(df) < 50:
        return {
            'levels': {},
            'current_zone': "Insufficient data for Fibonacci analysis",
            'current_price': 0,
            'swing_high': 0,
            'swing_low': 0,
            'retracement_level': "N/A",
            'zone_description': ""
        }
    
    try:
        # Use last 50 days for more accurate swing points
        recent_data = df.tail(50)
        closes = recent_data['close'].values
        
        # Find swing high and low
        swing_high = float(np.max(closes))
        swing_low = float(np.min(closes))
        diff = swing_high - swing_low
        
        if diff == 0:
            return {
                'levels': {},
                'current_zone': "No price movement detected",
                'current_price': float(closes[-1]),
                'swing_high': swing_high,
                'swing_low': swing_low,
                'retracement_level': "N/A",
                'zone_description': ""
            }
        
        # Fibonacci levels
        fib_levels = {
            '0.0': swing_high,
            '0.236': swing_high - diff * 0.236,
            '0.382': swing_high - diff * 0.382,
            '0.5': swing_high - diff * 0.5,
            '0.618': swing_high - diff * 0.618,
            '0.786': swing_high - diff * 0.786,
            '1.0': swing_low
        }
        
        current_price = float(closes[-1])
        
        # Determine Fibonacci zone
        zone_info = determine_fibonacci_zone(current_price, fib_levels)
        
        return {
            'levels': {k: float(v) for k, v in fib_levels.items()},
            'current_zone': zone_info['zone'],
            'current_price': current_price,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'retracement_level': zone_info['level'],
            'zone_description': zone_info['description'],
            'distance_from_high_pct': round(((swing_high - current_price) / diff) * 100, 2),
            'distance_from_low_pct': round(((current_price - swing_low) / diff) * 100, 2)
        }
        
    except Exception as e:
        print(f"Fibonacci calculation error: {e}")
        return {
            'levels': {},
            'current_zone': "Calculation error",
            'current_price': 0,
            'swing_high': 0,
            'swing_low': 0,
            'retracement_level': "N/A",
            'zone_description': ""
        }

def determine_fibonacci_zone(price: float, fib_levels: Dict) -> Dict:
    """Determine which Fibonacci zone the price is in"""
    levels = list(fib_levels.items())
    levels.sort(key=lambda x: x[1], reverse=True)  # Sort descending
    
    for i in range(len(levels)-1):
        level_name, level_value = levels[i]
        next_level_name, next_level_value = levels[i+1]
        
        if price >= next_level_value and price <= level_value:
            # Special handling for key Fibonacci levels
            if level_name in ['0.382', '0.5', '0.618']:
                return {
                    'zone': f"Fibonacci {level_name} - {next_level_name} Retracement Zone",
                    'level': f"Between {level_name} and {next_level_name}",
                    'description': get_fibonacci_description(level_name, next_level_name)
                }
            elif level_name == '0.0':
                return {
                    'zone': "Near Swing High",
                    'level': "0.0",
                    'description': "Price is near recent highs, potential resistance"
                }
            elif next_level_name == '1.0':
                return {
                    'zone': "Near Swing Low",
                    'level': "1.0",
                    'description': "Price is near recent lows, potential support"
                }
    
    return {
        'zone': "Outside Fibonacci Grid",
        'level': "N/A",
        'description': "Price is outside typical Fibonacci retracement levels"
    }

def get_fibonacci_description(level1: str, level2: str) -> str:
    """Get description for Fibonacci zone"""
    zones = {
        ('0.0', '0.236'): "Shallow retracement, strong trend continuation likely",
        ('0.236', '0.382'): "Moderate retracement, healthy pullback in trend",
        ('0.382', '0.5'): "Deep retracement, 50% level is critical support/resistance",
        ('0.5', '0.618'): "Golden ratio zone, most important Fibonacci level",
        ('0.618', '0.786'): "Deep retracement, trend may be weakening",
        ('0.786', '1.0'): "Very deep retracement, potential trend reversal"
    }
    return zones.get((level1, level2), "Standard Fibonacci retracement level")

def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """Detect common candlestick patterns without TA-Lib"""
    patterns = []
    
    try:
        if len(df) < 40:
            return ["Insufficient data for pattern recognition"]
        
        # Get recent data (last 40 days for pattern detection)
        recent_data = df.tail(40)
        
        # Extract OHLC data
        if 'open' in recent_data.columns and 'high' in recent_data.columns and 'low' in recent_data.columns:
            opens = recent_data['open'].values
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
        else:
            # Estimate if not available
            closes = recent_data['close'].values
            opens = closes * 0.99
            highs = closes * 1.01
            lows = closes * 0.99
        
        # Get last 5 days for pattern detection
        last_idx = len(closes) - 1
        
        if last_idx >= 4:  # Need at least 5 days
            # Bullish Engulfing Pattern
            if (closes[last_idx-1] < opens[last_idx-1] and  # Previous day is bearish
                closes[last_idx] > opens[last_idx] and      # Current day is bullish
                opens[last_idx] < closes[last_idx-1] and    # Open below previous close
                closes[last_idx] > opens[last_idx-1]):      # Close above previous open
                patterns.append("Bullish Engulfing Pattern")
            
            # Bearish Engulfing Pattern
            elif (closes[last_idx-1] > opens[last_idx-1] and  # Previous day is bullish
                  closes[last_idx] < opens[last_idx] and      # Current day is bearish
                  opens[last_idx] > closes[last_idx-1] and    # Open above previous close
                  closes[last_idx] < opens[last_idx-1]):      # Close below previous open
                patterns.append("Bearish Engulfing Pattern")
            
            # Hammer
            body = abs(closes[last_idx] - opens[last_idx])
            lower_shadow = min(opens[last_idx], closes[last_idx]) - lows[last_idx]
            upper_shadow = highs[last_idx] - max(opens[last_idx], closes[last_idx])
            
            if (lower_shadow > body * 2 and  # Long lower shadow
                upper_shadow < body * 0.1 and  # Small or no upper shadow
                closes[last_idx] > opens[last_idx]):  # Bullish close
                patterns.append("Bullish Hammer")
            
            # Hanging Man (similar to hammer but after uptrend)
            if (lower_shadow > body * 2 and  # Long lower shadow
                upper_shadow < body * 0.1 and  # Small or no upper shadow
                closes[last_idx] < opens[last_idx] and  # Bearish close
                closes[last_idx-5:last_idx].mean() < closes[last_idx]):  # In uptrend
                patterns.append("Bearish Hanging Man")
            
            # Doji
            if body < (highs[last_idx] - lows[last_idx]) * 0.1:  # Very small body
                patterns.append("Doji Pattern")
            
            # Shooting Star
            if (upper_shadow > body * 2 and  # Long upper shadow
                lower_shadow < body * 0.1 and  # Small or no lower shadow
                closes[last_idx] < opens[last_idx]):  # Bearish close
                patterns.append("Bearish Shooting Star")
            
            # Inverted Hammer
            if (upper_shadow > body * 2 and  # Long upper shadow
                lower_shadow < body * 0.1 and  # Small or no lower shadow
                closes[last_idx] > opens[last_idx]):  # Bullish close
                patterns.append("Bullish Inverted Hammer")
        
        # Multi-day patterns (need at least 3 days)
        if last_idx >= 2:
            # Morning Star (3-day pattern)
            if (closes[last_idx-2] < opens[last_idx-2] and  # Day 1: bearish
                abs(closes[last_idx-1] - opens[last_idx-1]) < (highs[last_idx-1] - lows[last_idx-1]) * 0.3 and  # Day 2: small body
                closes[last_idx] > opens[last_idx] and  # Day 3: bullish
                closes[last_idx] > (opens[last_idx-2] + closes[last_idx-2]) / 2):  # Closes above midpoint of Day 1
                patterns.append("Bullish Morning Star")
            
            # Evening Star (3-day pattern)
            if (closes[last_idx-2] > opens[last_idx-2] and  # Day 1: bullish
                abs(closes[last_idx-1] - opens[last_idx-1]) < (highs[last_idx-1] - lows[last_idx-1]) * 0.3 and  # Day 2: small body
                closes[last_idx] < opens[last_idx] and  # Day 3: bearish
                closes[last_idx] < (opens[last_idx-2] + closes[last_idx-2]) / 2):  # Closes below midpoint of Day 1
                patterns.append("Bearish Evening Star")
            
            # Three White Soldiers (3 consecutive bullish candles)
            if all(closes[i] > opens[i] for i in range(last_idx-2, last_idx+1)):
                if (closes[last_idx-2] > opens[last_idx-2] and
                    closes[last_idx-1] > closes[last_idx-2] and
                    closes[last_idx] > closes[last_idx-1]):
                    patterns.append("Bullish Three White Soldiers")
            
            # Three Black Crows (3 consecutive bearish candles)
            if all(closes[i] < opens[i] for i in range(last_idx-2, last_idx+1)):
                if (closes[last_idx-2] < opens[last_idx-2] and
                    closes[last_idx-1] < closes[last_idx-2] and
                    closes[last_idx] < closes[last_idx-1]):
                    patterns.append("Bearish Three Black Crows")
            
            # Harami Pattern (2-day)
            body_day1 = abs(closes[last_idx-1] - opens[last_idx-1])
            body_day2 = abs(closes[last_idx] - opens[last_idx])
            
            if body_day1 > body_day2 * 2:  # Day 1 has much larger body
                # Bullish Harami
                if (closes[last_idx-1] < opens[last_idx-1] and  # Day 1 bearish
                    closes[last_idx] > opens[last_idx] and      # Day 2 bullish
                    opens[last_idx] > closes[last_idx-1] and    # Day 2 open above Day 1 close
                    closes[last_idx] < opens[last_idx-1]):      # Day 2 close below Day 1 open
                    patterns.append("Bullish Harami Pattern")
                # Bearish Harami
                elif (closes[last_idx-1] > opens[last_idx-1] and  # Day 1 bullish
                      closes[last_idx] < opens[last_idx] and      # Day 2 bearish
                      opens[last_idx] < closes[last_idx-1] and    # Day 2 open below Day 1 close
                      closes[last_idx] > opens[last_idx-1]):      # Day 2 close above Day 1 open
                    patterns.append("Bearish Harami Pattern")
            
            # Piercing Pattern (2-day)
            if (closes[last_idx-1] < opens[last_idx-1] and  # Day 1 bearish
                closes[last_idx] > opens[last_idx] and      # Day 2 bullish
                opens[last_idx] < closes[last_idx-1] and    # Day 2 open below Day 1 close
                closes[last_idx] > (opens[last_idx-1] + closes[last_idx-1]) / 2):  # Closes above midpoint
                patterns.append("Bullish Piercing Pattern")
            
            # Dark Cloud Cover (2-day)
            if (closes[last_idx-1] > opens[last_idx-1] and  # Day 1 bullish
                closes[last_idx] < opens[last_idx] and      # Day 2 bearish
                opens[last_idx] > closes[last_idx-1] and    # Day 2 open above Day 1 close
                closes[last_idx] < (opens[last_idx-1] + closes[last_idx-1]) / 2):  # Closes below midpoint
                patterns.append("Bearish Dark Cloud Cover")
        
        # Add chart patterns
        patterns.extend(detect_chart_patterns(df))
        
        return patterns if patterns else ["No strong candlestick patterns detected"]
        
    except Exception as e:
        print(f"Pattern detection error: {e}")
        return ["Pattern recognition unavailable"]

def detect_chart_patterns(df: pd.DataFrame) -> List[str]:
    """Detect chart patterns"""
    patterns = []
    
    try:
        if len(df) < 20:
            return []
        
        closes = df['close'].values
        
        # Detect wedges
        if len(df) >= 20:
            # Simple wedge detection (price converging)
            recent_closes = closes[-20:]
            high_trend = np.polyfit(range(20), df['high'].values[-20:] if 'high' in df.columns else recent_closes, 1)[0]
            low_trend = np.polyfit(range(20), df['low'].values[-20:] if 'low' in df.columns else recent_closes * 0.98, 1)[0]
            
            if high_trend < 0 and low_trend > 0:  # Converging
                patterns.append("Rising Wedge" if np.mean(recent_closes[-5:]) > np.mean(recent_closes[:5]) else "Falling Wedge")
        
        # Detect double top/bottom
        if len(df) >= 30:
            last_30 = closes[-30:]
            peaks = []
            troughs = []
            
            for i in range(1, 29):
                if last_30[i] > last_30[i-1] and last_30[i] > last_30[i+1]:
                    peaks.append(last_30[i])
                elif last_30[i] < last_30[i-1] and last_30[i] < last_30[i+1]:
                    troughs.append(last_30[i])
            
            if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.02:
                patterns.append("Double Top" if peaks[-1] < peaks[-2] else "Double Bottom Reversal")
            if len(troughs) >= 2 and abs(troughs[-1] - troughs[-2]) / troughs[-1] < 0.02:
                patterns.append("Double Bottom" if troughs[-1] > troughs[-2] else "Double Top Reversal")
        
        return patterns
        
    except Exception as e:
        print(f"Chart pattern detection error: {e}")
        return []

def calculate_support_resistance(df: pd.DataFrame) -> Dict:
    """Calculate support and resistance levels"""
    if len(df) < 30:
        return {
            'support': 0,
            'resistance': 0,
            'support_pivot': 0,
            'resistance_pivot': 0,
            'closest_level': "Insufficient data",
            'strength': "Weak"
        }
    
    try:
        # Use pivot point calculation
        recent = df.tail(30)
        high = float(recent['high'].max() if 'high' in recent.columns else recent['close'].max())
        low = float(recent['low'].min() if 'low' in recent.columns else recent['close'].min())
        close = float(recent['close'].iloc[-1])
        
        # Pivot points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        current_price = float(df['close'].iloc[-1])
        
        # Determine closest level
        levels = {
            'Resistance 1': r1,
            'Pivot': pivot,
            'Support 1': s1
        }
        
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
        
        return {
            'support': round(s1, 2),
            'resistance': round(r1, 2),
            'support_pivot': round(pivot - (r1 - s1) * 0.5, 2),
            'resistance_pivot': round(pivot + (r1 - s1) * 0.5, 2),
            'closest_level': closest_level[0],
            'distance_pct': round(abs(closest_level[1] - current_price) / current_price * 100, 2),
            'strength': "Strong" if abs(closest_level[1] - current_price) / current_price < 0.02 else "Moderate"
        }
        
    except Exception as e:
        print(f"Support/resistance calculation error: {e}")
        return {
            'support': 0,
            'resistance': 0,
            'support_pivot': 0,
            'resistance_pivot': 0,
            'closest_level': "Calculation error",
            'strength': "Unknown"
        }
    

def sync_historical_data(symbol: str, api_symbol: str) -> Dict:
    """Internal function to sync historical data"""
    try:
        ticker = yf.Ticker(api_symbol, session=session)
        
        # For crypto, try to get OHLC data
        try:
            hist = ticker.history(period="2y", interval="1d")
        except:
            # Fallback to 1 year if 2 years fails
            hist = ticker.history(period="1y", interval="1d")
        
        new_data = []
        for date, row in hist.iterrows():
            new_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": float(round(row['Open'], 2)),
                "high": float(round(row['High'], 2)),
                "low": float(round(row['Low'], 2)),
                "close": float(round(row['Close'], 2)),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        # Update Firebase with new data
        asset_ref = db.collection('historical_data').document(api_symbol)
        asset_ref.set({
            "daily": new_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(new_data)
        }, merge=False)
        
        return {
            "success": True,
            "data_points": len(new_data),
            "message": f"Synced {len(new_data)} days of data for {symbol}"
        }
        
    except Exception as e:
        print(f"Sync error: {e}")
        return {"success": False, "error": str(e)}

# ==================== ENDPOINTS ====================

@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        api_symbol = get_api_symbol(symbol)
        ticker = yf.Ticker(api_symbol, session=session)
        
        # 1. Attempt download
        hist = ticker.history(period="2y", interval="1d")
        
        # 2. CRITICAL CHECK: If hist is empty, Yahoo has blocked the request
        if hist.empty:
            return jsonify({
                "status": "error",
                "message": "Yahoo Finance returned no data. You are likely being rate-limited.",
                "symbol": symbol
            }), 503  # Service Unavailable

        new_data = []
        for date, row in hist.iterrows():
            new_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": float(round(row['Open'], 2)),
                "high": float(round(row['High'], 2)),
                "low": float(round(row['Low'], 2)),
                "close": float(round(row['Close'], 2)),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        # 3. Only save to Firebase if we actually have data
        asset_ref = db.collection('historical_data').document(api_symbol)
        asset_ref.set({
            "daily": new_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(new_data)
        }, merge=False)
        
        return jsonify({
            "status": "synced",
            "symbol": symbol,
            "data_points": len(new_data),
            "period": f"{len(new_data)} days"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    """Main AI analysis endpoint with all technical indicators"""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    interval = request.args.get('interval', 'daily')
    
    # Determine window size
    if 'monthly' in interval.lower():
        interval_type = 'monthly'
        window_days = 1460  # 4 years in days
        description = "Macro Analysis (4 Years)"
    else:
        interval_type = 'daily'
        window_days = 180  # 6 months
        description = "Swing Analysis (6 Months)"
    
    api_symbol = get_api_symbol(symbol)
    
    try:
        # Get data from Firebase
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        
        # If no data exists, fetch it first
        if not asset_doc.exists or not asset_doc.to_dict().get('daily'):
            # Call the sync function internally
            sync_result = sync_historical_data(symbol, api_symbol)
            if not sync_result.get('success', False):
                return jsonify({
                    "error": f"Failed to fetch historical data for {symbol}",
                    "message": "Please try again or use /get-historical-data first"
                }), 500
            
            # Re-fetch the document
            asset_doc = db.collection('historical_data').document(api_symbol).get()
        
        data_dict = asset_doc.to_dict()
        full_data = data_dict.get('daily', [])
        
        if not full_data:
            return jsonify({"error": "No daily data available after sync. Please try again."}), 404
        # Convert to DataFrame
        df = pd.DataFrame(full_data)
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            return jsonify({"error": "Invalid data format"}), 500
        
        # Use appropriate window
        analysis_df = df.tail(min(window_days, len(df))).copy()
        
        if len(analysis_df) < 30:
            return jsonify({"error": f"Insufficient data. Only {len(analysis_df)} days available. Need at least 30 days."}), 400
        
        # Calculate all technical indicators
        ma_data = calculate_moving_averages(analysis_df)
        fib_data = calculate_fibonacci_retracement(analysis_df)
        sr_data = calculate_support_resistance(analysis_df)
        patterns = detect_candlestick_patterns(analysis_df)
        
        # Get news from Alpha Vantage
        news_items = get_alpha_vantage_news(symbol)
        news_titles = [item['title'] for item in news_items[:5]]
        news_sentiment = "neutral"
        if news_items:
            avg_sentiment = sum(item['sentiment_score'] for item in news_items) / len(news_items)
            if avg_sentiment > 0.1:
                news_sentiment = "positive"
            elif avg_sentiment < -0.1:
                news_sentiment = "negative"
        
        # Get current price from yfinance for accuracy
        try:
            ticker = yf.Ticker(api_symbol, session=session)
            current_price = float(ticker.fast_info.get('last_price', analysis_df['close'].iloc[-1]))
        except:
            current_price = float(analysis_df['close'].iloc[-1])
        
        # Prepare data for AI
        technical_summary = {
            "symbol": symbol,
            "analysis_type": description,
            "current_price": current_price,
            "price_formatted": format_currency(current_price),
            "data_period": f"{len(analysis_df)} days",
            "moving_averages": {
                "MA20": format_currency(ma_data.get('MA20', 0)),
                "MA50": format_currency(ma_data.get('MA50', 0)),
                "MA200": format_currency(ma_data.get('MA200', 0)),
                "trend": ma_data.get('trend', 'Unknown'),
                "alignment": ma_data.get('ma_alignment', 'Unknown'),
                "golden_cross": ma_data.get('golden_cross', 'Unknown')
            },
            "fibonacci": {
                "zone": fib_data.get('current_zone', 'Unknown'),
                "retracement_level": fib_data.get('retracement_level', 'N/A'),
                "description": fib_data.get('zone_description', ''),
                "swing_high": format_currency(fib_data.get('swing_high', 0)),
                "swing_low": format_currency(fib_data.get('swing_low', 0)),
                "current_vs_high": f"{fib_data.get('distance_from_high_pct', 0)}% below high",
                "current_vs_low": f"{fib_data.get('distance_from_low_pct', 0)}% above low"
            },
            "support_resistance": {
                "support": format_currency(sr_data.get('support', 0)),
                "resistance": format_currency(sr_data.get('resistance', 0)),
                "closest_level": sr_data.get('closest_level', 'Unknown'),
                "distance": f"{sr_data.get('distance_pct', 0)}%",
                "strength": sr_data.get('strength', 'Unknown')
            },
            "patterns_detected": patterns,
            "news": {
                "sentiment": news_sentiment,
                "headlines": news_titles,
                "count": len(news_titles)
            }
        }
        
        # Create comprehensive prompt for AI
        prompt = f"""
        Perform a comprehensive technical analysis for {symbol} based on the following data:
        
        ANALYSIS TYPE: {description}
        CURRENT PRICE: {technical_summary['price_formatted']}
        DATA PERIOD: {technical_summary['data_period']}
        
        MOVING AVERAGES:
        - MA20: {technical_summary['moving_averages']['MA20']}
        - MA50: {technical_summary['moving_averages']['MA50']}
        - MA200: {technical_summary['moving_averages']['MA200']}
        - Trend: {technical_summary['moving_averages']['trend']}
        - Alignment: {technical_summary['moving_averages']['alignment']}
        - Golden Cross: {technical_summary['moving_averages']['golden_cross']}
        
        FIBONACCI RETRACEMENT:
        - Current Zone: {technical_summary['fibonacci']['zone']}
        - Retracement Level: {technical_summary['fibonacci']['retracement_level']}
        - Description: {technical_summary['fibonacci']['description']}
        - Swing High: {technical_summary['fibonacci']['swing_high']}
        - Swing Low: {technical_summary['fibonacci']['swing_low']}
        - Position: {technical_summary['fibonacci']['current_vs_high']}, {technical_summary['fibonacci']['current_vs_low']}
        
        SUPPORT & RESISTANCE:
        - Support: {technical_summary['support_resistance']['support']}
        - Resistance: {technical_summary['support_resistance']['resistance']}
        - Closest Level: {technical_summary['support_resistance']['closest_level']} ({technical_summary['support_resistance']['distance']} away)
        - Strength: {technical_summary['support_resistance']['strength']}
        
        PATTERNS DETECTED:
        {chr(10).join([f"- {pattern}" for pattern in patterns])}
        
        NEWS SENTIMENT: {technical_summary['news']['sentiment'].upper()}
        RECENT HEADLINES ({technical_summary['news']['count']}):
        {chr(10).join([f"- {headline}" for headline in technical_summary['news']['headlines']])}
        
        Based on ALL technical indicators above, provide a comprehensive analysis including:
        1. Overall trend assessment with confidence level
        2. Significance of Fibonacci retracement level
        3. Moving average alignment implications
        4. Pattern recognition analysis
        5. Support/resistance breakout potential
        6. News sentiment impact
        7. Risk assessment with specific risk factors
        8. Clear trading recommendation with entry/exit levels
        
        Respond in valid JSON format with these exact fields:
        - trend: (string) Overall market trend with confidence
        - patterns: (array of strings) Key technical patterns with explanations
        - fibonacci_analysis: (string) Detailed Fibonacci interpretation
        - ma_analysis: (string) Moving average analysis with implications
        - support_resistance_analysis: (string) Breakout/breakdown potential
        - news_impact: (string) How news affects the analysis
        - risk_assessment: (string) Specific risk factors (Low/Medium/High)
        - verdict: (string) Comprehensive analysis conclusion
        - suggestion: (string) BUY/HOLD/SELL with conviction
        - confidence_score: (number 0-100)
        - recommended_action: (string) Specific action with price targets
        - key_levels: (object) Important price levels to watch
        """
        
        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional technical analyst with 20 years experience. Provide detailed, actionable insights. Respond ONLY in valid JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        
        # Combine all data
        result = {
            **ai_response,
            "technical_summary": technical_summary,
            "analysis_metadata": {
                "symbol": symbol,
                "interval": interval_type,
                "analysis_date": datetime.now().isoformat(),
                "data_points": len(analysis_df)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"AI insight error: {e}")
        return jsonify({"error": str(e), "message": "Analysis failed. Please try syncing data first."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "alpha_vantage_available": bool(ALPHA_VANTAGE_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
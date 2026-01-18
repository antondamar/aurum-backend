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

def get_alpha_vantage_historical(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Get historical data from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    try:
        # Map period to Alpha Vantage output size
        output_size_map = {
            "1y": "compact",
            "2y": "full",
            "5y": "full",
            "10y": "full"
        }
        
        output_size = output_size_map.get(period, "full")
        
        # Alpha Vantage TIME_SERIES_DAILY endpoint
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = session.get(url, timeout=10)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            # Check for error or rate limit
            if 'Note' in data or 'Information' in data:
                print(f"Alpha Vantage rate limit or info: {data.get('Note', data.get('Information', 'Unknown'))}")
                return None
            print(f"No time series data in response for {symbol}")
            return None
        
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame
        records = []
        for date_str, values in time_series.items():
            records.append({
                'date': date_str,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(float(values['5. volume']))
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
        
    except Exception as e:
        print(f"Alpha Vantage historical data error for {symbol}: {e}")
        return None

def get_alpha_vantage_news(symbol: str) -> List[Dict]:
    """Get news from Alpha Vantage with better error handling"""
    if not ALPHA_VANTAGE_API_KEY:
        print("⚠️ No Alpha Vantage API key")
        return []
    
    try:
        # Clean symbol for Alpha Vantage
        clean_symbol = symbol.upper()
        if clean_symbol.endswith('.JK'):
            clean_symbol = clean_symbol.replace('.JK', '')
        if clean_symbol.endswith('-USD'):
            clean_symbol = clean_symbol.replace('-USD', '')
        
        print(f"📰 Fetching news for {clean_symbol}...")
        
        # Alpha Vantage news endpoint
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={clean_symbol}&apikey={ALPHA_VANTAGE_API_KEY}&limit=5"
        
        response = session.get(url, timeout=15)
        
        # Check for rate limiting
        if 'Note' in response.text or 'Information' in response.text:
            print(f"⚠️ Alpha Vantage rate limited for {symbol}")
            return []
        
        data = response.json()
        
        # Debug: print raw response
        print(f"News API response keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
        
        news_items = []
        if 'feed' in data and data['feed']:
            for item in data['feed'][:3]:  # Get top 3 news
                # Extract ticker sentiments
                ticker_sentiments = item.get('ticker_sentiment', [])
                symbol_sentiment = 0
                
                # Find sentiment for our symbol
                for ticker_info in ticker_sentiments:
                    if ticker_info.get('ticker') == clean_symbol:
                        symbol_sentiment = float(ticker_info.get('ticker_sentiment_score', 0))
                        break
                
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'time_published': item.get('time_published', ''),
                    'url': item.get('url', ''),
                    'sentiment_score': symbol_sentiment,
                    'overall_sentiment': float(item.get('overall_sentiment_score', 0)),
                    'sentiment_label': item.get('overall_sentiment_label', 'neutral')
                })
            
            print(f"✅ Found {len(news_items)} news items for {symbol}")
        else:
            print(f"⚠️ No news found for {symbol}")
            # Return mock news for testing
            news_items = get_mock_news(symbol)
        
        return news_items
        
    except Exception as e:
        print(f"❌ Alpha Vantage news error for {symbol}: {e}")
        # Return mock data as fallback
        return get_mock_news(symbol)

def get_mock_news(symbol: str) -> List[Dict]:
    """Provide mock news when API fails"""
    return [
        {
            'title': f'{symbol} shows strong technical patterns',
            'summary': f'Technical analysis indicates potential breakout for {symbol}',
            'source': 'Technical Analysis',
            'time_published': datetime.now().strftime('%Y%m%dT%H%M%S'),
            'sentiment_score': 0.15,
            'sentiment_label': 'positive'
        },
        {
            'title': f'Market watching {symbol} closely',
            'summary': f'Traders are monitoring {symbol} for upcoming trends',
            'source': 'Market Watch',
            'time_published': datetime.now().strftime('%Y%m%dT%H%M%S'),
            'sentiment_score': 0.05,
            'sentiment_label': 'neutral'
        }
    ]

def get_yfinance_historical(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Get historical data from yfinance (fallback)"""
    try:
        api_symbol = get_api_symbol(symbol)
        ticker = yf.Ticker(api_symbol, session=session)
        
        # Try multiple periods
        for p in [period, "1y", "6mo", "3mo", "1mo"]:
            try:
                hist = ticker.history(period=p, interval="1d")
                if not hist.empty:
                    # Convert to our format
                    df = pd.DataFrame({
                        'date': hist.index,
                        'open': hist['Open'].values,
                        'high': hist['High'].values,
                        'low': hist['Low'].values,
                        'close': hist['Close'].values,
                        'volume': hist['Volume'].values if 'Volume' in hist.columns else 0
                    })
                    return df
            except Exception as e:
                print(f"yfinance error for period {p}: {e}")
                continue
        
        return None
        
    except Exception as e:
        print(f"yfinance historical data error for {symbol}: {e}")
        return None

# ==================== SYNC FUNCTION ====================

def sync_historical_data(symbol: str) -> Dict:
    """Sync historical data with Alpha Vantage as primary, yfinance as fallback"""
    try:
        print(f"Starting sync for {symbol}")
        
        # Try Alpha Vantage first
        df = get_alpha_vantage_historical(symbol, "2y")
        source = "alpha_vantage"
        
        # If Alpha Vantage fails, try yfinance
        if df is None or df.empty:
            print(f"Alpha Vantage failed for {symbol}, trying yfinance...")
            df = get_yfinance_historical(symbol, "2y")
            source = "yfinance"
        
        # If both fail, return error
        if df is None or df.empty:
            return {
                "success": False,
                "error": "Could not fetch data from any source",
                "source": "none"
            }
        
        # Prepare data for Firebase
        new_data = []
        for _, row in df.iterrows():
            new_data.append({
                "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                "open": float(round(row['open'], 2)),
                "high": float(round(row['high'], 2)),
                "low": float(round(row['low'], 2)),
                "close": float(round(row['close'], 2)),
                "volume": int(row['volume']) if 'volume' in row else 0
            })
        
        # Update Firebase
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        asset_ref.set({
            "daily": new_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(new_data),
            "data_source": source
        }, merge=False)
        
        print(f"Synced {len(new_data)} days of data for {symbol} from {source}")
        
        return {
            "success": True,
            "data_points": len(new_data),
            "source": source,
            "message": f"Synced {len(new_data)} days of data for {symbol} from {source}"
        }
        
    except Exception as e:
        print(f"Sync error for {symbol}: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "error"
        }

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

def downsample_to_50_points(df: pd.DataFrame) -> pd.DataFrame:
    """Always reduces the dataframe to exactly 50 representative points"""
    if len(df) <= 50:
        return df
    # Logic: Total length / 50 = step size
    step = len(df) // 50
    # Select every nth row to maintain consistent trend representation
    downsampled = df.iloc[::step].tail(50) 
    return downsampled

# ==================== ENDPOINTS ====================

@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    """Sync historical data endpoint"""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    
    try:
        # Use the new sync function
        sync_result = sync_historical_data(symbol)
        
        if not sync_result.get('success', False):
            return jsonify({
                "status": "error",
                "symbol": symbol,
                "error": sync_result.get('error', 'Unknown error'),
                "source": sync_result.get('source', 'unknown'),
                "message": "Failed to fetch data from any source"
            }), 503
        
        return jsonify({
            "status": "synced",
            "symbol": symbol,
            "data_points": sync_result['data_points'],
            "source": sync_result['source'],
            "period": f"{sync_result['data_points']} days",
            "message": sync_result['message']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval', 'daily')
    timeframe = request.args.get('timeframe', '2y')
    
    api_symbol = get_api_symbol(symbol)
    
    # 1. DEFINE MISSING METADATA VARIABLES
    description = "Macro (4Y) Portfolio Strategy" if interval == 'monthly' else "Swing (6M) Technical Analysis"
    interval_type = "Monthly" if interval == 'monthly' else "Daily"
    
    try:
        # 2. Fetch requested timeframe from Firebase or API
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        # FIX: Ensure data_dict is defined for the response
        if asset_doc.exists:
            data_dict = asset_doc.to_dict()
        else:
            sync_result = sync_historical_data(symbol)
            if not sync_result.get('success'):
                return jsonify({"error": "Data sync failed", "message": sync_result.get('error')}), 503
            data_dict = db.collection('historical_data').document(api_symbol).get().to_dict()
        
        # Determine data pool
        data_pool = data_dict.get('monthly', []) if interval == 'monthly' else data_dict.get('daily', [])
        if not data_pool:
            # Fallback if specific interval list is empty
            data_pool = data_dict.get('daily', [])
            
        df = pd.DataFrame(data_pool)
        
        # 3. Filter by timeframe
        df['date'] = pd.to_datetime(df['date'])
        years_to_keep = int(timeframe.replace('y', ''))
        cutoff_date = datetime.now() - timedelta(days=years_to_keep * 365)
        df = df[df['date'] >= cutoff_date]

        # 4. Calculate Moving Averages on FULL timeframe (Fixes MA=0 issue)
        ma_data = calculate_moving_averages(df)

        # 5. DOWNSAMPLE TO EXACTLY 50 POINTS for AI context
        analysis_df = downsample_to_50_points(df)
        
        # 6. Perform Technical Analysis on these 50 points
        fib_data = calculate_fibonacci_retracement(analysis_df)
        sr_data = calculate_support_resistance(analysis_df)
        # FIX: Define patterns variable
        patterns = detect_candlestick_patterns(analysis_df) or ["No patterns detected"]
        
        # 7. Fetch News
        news_items = get_alpha_vantage_news(symbol) or get_mock_news(symbol)
        news_titles = [item['title'] for item in news_items[:3]]

        news_sentiment = "neutral"
        sentiment_score = 0

        if news_items:
            scores = [item.get('sentiment_score', 0) for item in news_items]
            valid_scores = [s for s in scores if s != 0]
            if valid_scores:
                avg_sentiment = sum(valid_scores) / len(valid_scores)
                sentiment_score = avg_sentiment
                if avg_sentiment > 0.15: news_sentiment = "bullish"
                elif avg_sentiment < -0.15: news_sentiment = "bearish"
        
        current_price = float(analysis_df['close'].iloc[-1])
        
        # 8. Prepare Technical Summary with Correct Variables
        technical_summary = {
            "symbol": symbol,
            "analysis_type": description,
            "current_price": current_price,
            "price_formatted": format_currency(current_price),
            "data_period": f"{len(analysis_df)} points (downsampled)",
            "data_source": data_dict.get('data_source', 'unknown'),
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
                "sentiment_score": round(sentiment_score, 3),
                "headlines": news_titles,
                "count": len(news_titles),
                "source": "Alpha Vantage News Sentiment API"
            }
        }
        # Create comprehensive prompt for AI
        prompt = f"""
        Perform a comprehensive technical analysis for {symbol} based on the following data:
        
        ANALYSIS TYPE: {description}
        CURRENT PRICE: {technical_summary['price_formatted']}
        DATA PERIOD: {technical_summary['data_period']}
        DATA SOURCE: {technical_summary['data_source'].upper()}
        
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
        
        NEWS SENTIMENT: {technical_summary['news']['sentiment'].upper()} (Score: {technical_summary['news']['sentiment_score']})
        SENTIMENT SOURCE: {technical_summary['news']['source']}
        RECENT HEADLINES ({technical_summary['news']['count']}):
        {chr(10).join([f"- {headline}" for headline in technical_summary['news']['headlines']])}
        
        Based on ALL technical indicators above, provide a comprehensive analysis including:
        1. Overall trend assessment with confidence level
        2. Significance of Fibonacci retracement level
        3. Moving average alignment implications
        4. Pattern recognition analysis
        5. Support/resistance breakout potential
        6. News sentiment impact with quantifiable score
        7. Risk assessment with specific risk factors
        8. Clear trading recommendation with entry/exit levels
        
        Respond in valid JSON format with these exact fields:
        - trend: (string) Overall market trend with confidence
        - patterns: (array of strings) Key technical patterns with explanations
        - fibonacci_analysis: (string) Detailed Fibonacci interpretation
        - ma_analysis: (string) Moving average analysis with implications
        - support_resistance_analysis: (string) Breakout/breakdown potential
        - news_impact: (string) How news sentiment affects the analysis
        - risk_assessment: (string) Specific risk factors (Low/Medium/High)
        - verdict: (string) Comprehensive analysis conclusion
        - suggestion: (string) BUY/HOLD/SELL with conviction
        - confidence_score: (number 0-100)
        - recommended_action: (string) Specific action with price targets
        - key_levels: (object) Important price levels to watch

        Additional Rules: 
        - Summarizze risk assessment up to 15 words response.
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
                "data_points": len(analysis_df),
                "data_source": technical_summary['data_source']
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
        "alpha_vantage_available": bool(ALPHA_VANTAGE_API_KEY),
        "alpha_vantage_key_set": bool(ALPHA_VANTAGE_API_KEY)
    })

# ==================== DIRECT UPDATE ENDPOINT ====================
@app.route('/direct-update', methods=['POST'])
def direct_update():
    """Simple endpoint to accept data from local sync"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        historical_data = data.get('data', [])
        
        if not symbol or not historical_data:
            return jsonify({"error": "Missing symbol or data"}), 400
        
        # Store in Firebase
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        
        asset_ref.set({
            "daily": historical_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(historical_data),
            "data_source": "local_sync"
        }, merge=False)
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "data_points": len(historical_data),
            "message": f"Updated {len(historical_data)} days of data"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ==================== LOCAL UPLOAD ENDPOINT ====================
@app.route('/local-upload', methods=['POST'])
def local_upload():
    """Endpoint for laptop to upload data - APPENDS new data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        new_historical_data = data.get('data', [])
        
        if not symbol or not new_historical_data:
            return jsonify({"error": "Missing symbol or data"}), 400
        
        # Get existing data from Firebase
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()
        
        existing_data = []
        if doc.exists:
            existing_dict = doc.to_dict()
            existing_data = existing_dict.get('daily', [])
        
        # Merge: Keep existing, add new unique dates
        existing_dates = {item['date'] for item in existing_data}
        merged_data = existing_data.copy()
        
        new_items_added = 0
        for new_item in new_historical_data:
            if new_item['date'] not in existing_dates:
                merged_data.append(new_item)
                new_items_added += 1
        
        # Sort by date
        merged_data.sort(key=lambda x: x['date'])
        
        # Update Firebase
        asset_ref.set({
            "daily": merged_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(merged_data),
            "data_source": f"local_upload (merged, added {new_items_added} new)"
        }, merge=False)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "existing_points": len(existing_data),
            "new_points_added": new_items_added,
            "total_points": len(merged_data),
            "message": f"Merged data: {len(existing_data)} existing + {new_items_added} new = {len(merged_data)} total"
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
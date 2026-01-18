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

# ==================== FETCH REAL-TIME PRICE ====================

def get_realtime_price(symbol: str) -> Optional[float]:
    """Fetch real-time price using yfinance (matches priceService.js logic)"""
    try:
        api_symbol = get_api_symbol(symbol)
        ticker = yf.Ticker(api_symbol, session=session)
        
        # Try to get current price from multiple sources
        try:
            # Method 1: fast_info (fastest)
            if hasattr(ticker, 'fast_info'):
                price = ticker.fast_info.get('last_price') or ticker.fast_info.get('regularMarketPrice')
                if price and price > 0:
                    return float(price)
        except:
            pass
        
        try:
            # Method 2: info (more reliable but slower)
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price and price > 0:
                return float(price)
        except:
            pass
        
        try:
            # Method 3: history last close (fallback)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        return None
        
    except Exception as e:
        print(f"Real-time price fetch error for {symbol}: {e}")
        return None

# ==================== SYNC FUNCTION ====================

def sync_historical_data(symbol: str) -> Dict:
    """Smart sync that appends new data to existing history."""
    try:
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()
        
        existing_daily = []
        if doc.exists:
            data = doc.to_dict()
            existing_daily = data.get('daily', [])
            last_synced = datetime.fromisoformat(data.get('last_synced'))
            # 24-HOUR TIMER: Only sync if it's been a day
            if datetime.now() - last_synced < timedelta(hours=24):
                return {"success": True, "source": "firebase_cache", "message": "Using cached 24h data"}

        # FETCH ONLY RECENT DATA (1 Month) TO APPEND
        # This prevents re-downloading 10 years every time
        new_df = get_alpha_vantage_historical(symbol, "1mo")
        if new_df is None or new_df.empty:
            new_df = get_yfinance_historical(symbol, "1mo")
            
        if new_df is None or new_df.empty:
            return create_mock_historical_data(symbol)

        # CONVERT TO LIST OF DICTS
        new_points = new_df.to_dict('records')
        for p in new_points:
            p['date'] = p['date'].strftime('%Y-%m-%d') if not isinstance(p['date'], str) else p['date']

        # SMART MERGE: Combine, Sort, and Drop Duplicates
        all_data_df = pd.concat([pd.DataFrame(existing_daily), pd.DataFrame(new_points)])
        all_data_df = all_data_df.drop_duplicates(subset=['date']).sort_values('date')
        
        final_daily = all_data_df.to_dict('records')
        monthly_data = aggregate_to_monthly(final_daily)

        asset_ref.set({
            "daily": final_daily,
            "monthly": monthly_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_points": len(final_daily),
            "data_source": "hybrid_merge"
        }, merge=False)

        return {"success": True, "message": f"Appended new data to {symbol}"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_mock_historical_data(symbol: str) -> Dict:
    """Returns an error instead of fake data to prevent price inaccuracy."""
    return {
        "success": False, 
        "error": "API Rate Limit reached", 
        "message": f"Real-time data for {symbol} is currently unavailable. Please try again later."
    }

# ==================== TECHNICAL ANALYSIS FUNCTIONS ====================

def aggregate_to_monthly(daily_data: List[Dict]) -> List[Dict]:
    """Convert daily OHLC data to monthly"""
    monthly_data = {}
    for day in daily_data:
        date = pd.to_datetime(day['date'])
        month_key = f"{date.year}-{date.month:02d}"
        
        if month_key not in monthly_data:
            monthly_data[month_key] = {
                'date': f"{date.year}-{date.month:02d}-01",
                'open': day['open'],
                'high': day['high'],
                'low': day['low'],
                'close': day['close'],
                'volume': day['volume']
            }
        else:
            monthly_data[month_key]['high'] = max(monthly_data[month_key]['high'], day['high'])
            monthly_data[month_key]['low'] = min(monthly_data[month_key]['low'], day['low'])
            monthly_data[month_key]['close'] = day['close']
            monthly_data[month_key]['volume'] += day['volume']
    
    return sorted(monthly_data.values(), key=lambda x: x['date'])

def calculate_moving_averages(df: pd.DataFrame) -> Dict:
    """Calculate moving averages individually based on available data length"""
    ma_data = {}
    
    try:
        closes = pd.Series(df['close'].values)
        data_len = len(closes)
        current_price = float(closes.iloc[-1])
        
        # Define windows to check independently
        windows = [13, 20, 21, 50, 200]
        ma_values = {}
        
        for w in windows:
            if data_len >= w:
                # Calculate if we have enough points for this specific window
                val = float(closes.rolling(window=w).mean().iloc[-1])
                ma_values[f"MA{w}"] = round(val, 2)
            else:
                # Mark as N/A if history is too short for this timeframe
                ma_values[f"MA{w}"] = "N/A"
        
        # Determine trend based ONLY on AVAILABLE MAs
        active_mas = {k: v for k, v in ma_values.items() if v != "N/A"}
        
        if not active_mas:
            return {f"MA{w}": "N/A" for w in windows} | {"trend": "Insufficient Data", "ma_alignment": "N/A"}

        # Count price position relative to active MAs
        above_count = sum(1 for v in active_mas.values() if current_price > v)
        total_active = len(active_mas)
        
        # Dynamic Trend Logic: Percentage of active MAs the price is above
        ratio = above_count / total_active
        if ratio >= 0.8: trend = "Strong Bullish"
        elif ratio >= 0.6: trend = "Bullish"
        elif ratio <= 0.2: trend = "Strong Bearish"
        elif ratio <= 0.4: trend = "Bearish"
        else: trend = "Neutral"

        # Alignment check (check if MAs are stacked in order)
        alignment = "Mixed"
        sorted_keys = sorted(active_mas.keys(), key=lambda x: int(x[2:])) # Sort by window size
        active_vals = [active_mas[k] for k in sorted_keys]
        if active_vals == sorted(active_vals, reverse=True):
            alignment = "Bullish Stack"
        elif active_vals == sorted(active_vals):
            alignment = "Bearish Stack"

        ma_data = {
            **ma_values,
            'trend': trend,
            'golden_cross': "Yes" if (ma_values['MA50'] != "N/A" and ma_values['MA200'] != "N/A" and ma_values['MA50'] > ma_values['MA200']) else "No" if (ma_values['MA50'] != "N/A" and ma_values['MA200'] != "N/A") else "N/A",
            'ma_alignment': alignment,
            'ma_count_above': above_count,
            'total_active_mas': total_active
        }
        
        # Add directional flags for the AI context
        for w in windows:
            val = ma_values[f"MA{w}"]
            ma_data[f'price_vs_ma{w}'] = "Above" if val != "N/A" and current_price > val else "Below" if val != "N/A" else "N/A"

    except Exception as e:
        print(f"Moving average calculation error: {e}")
        ma_data = {f"MA{w}": "N/A" for w in [13, 20, 21, 50, 200]}
        ma_data['trend'] = "Calculation Error"
    
    return ma_data

def calculate_fibonacci_retracement(df: pd.DataFrame, interval: str = 'daily', current_price: Optional[float] = None) -> Dict:
    """Calculate Fibonacci retracement levels with interval-aware lookback"""
    
    # Interval-specific lookback periods
    if interval == 'monthly':
        lookback = min(12, len(df))  # Last 12 months for swing detection
        min_required = 6
    else:  # daily
        lookback = min(50, len(df))  # Last 50 days for swing detection
        min_required = 20
    
    if len(df) < min_required:
        return {
            'levels': {},
            'current_zone': f"Insufficient data for Fibonacci analysis ({len(df)} {interval} points)",
            'current_price': 0,
            'swing_high': 0,
            'swing_low': 0,
            'retracement_level': "N/A",
            'zone_description': ""
        }
    
    try:
        # Use interval-appropriate lookback period
        recent_data = df.tail(lookback)
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
        
        # Use provided current_price or fall back to last close
        price_to_use = current_price if current_price is not None else float(closes[-1])
        
        # Determine Fibonacci zone
        zone_info = determine_fibonacci_zone(price_to_use, fib_levels)
        
        return {
            'levels': {k: float(v) for k, v in fib_levels.items()},
            'current_zone': zone_info['zone'],
            'current_price': price_to_use,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'retracement_level': zone_info['level'],
            'zone_description': zone_info['description'],
            'distance_from_high_pct': round(((swing_high - price_to_use) / diff) * 100, 2),
            'distance_from_low_pct': round(((price_to_use - swing_low) / diff) * 100, 2),
            'lookback_period': f"{lookback} {interval} candles"
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

def detect_candlestick_patterns(df: pd.DataFrame, interval: str = 'daily') -> List[str]:
    """Detect common candlestick patterns without TA-Lib, with interval-specific thresholds"""
    patterns = []
    
    # Set thresholds based on interval
    if interval == 'monthly':
        # Monthly candles have larger price ranges, need more lenient thresholds
        body_ratio = 0.15      # For Doji: body < 15% of range (vs 10% for daily)
        shadow_ratio = 1.5     # For Hammer/Shooting Star: shadow > 1.5x body (vs 2x for daily)
        small_body_ratio = 0.25 # For Morning/Evening Star middle candle (vs 0.3 for daily)
        engulfing_ratio = 1.2   # For Harami: body_day1 > body_day2 * 1.2 (vs 2 for daily)
        trend_lookback = 8      # Look back 8 months for trend (vs 5 days for daily)
    else:  # daily
        body_ratio = 0.1       # Daily thresholds
        shadow_ratio = 2.0
        small_body_ratio = 0.3
        engulfing_ratio = 2.0
        trend_lookback = 5
    
    try:
        # Adjust minimum data points based on interval
        min_data_points = 20 if interval == 'monthly' else 40
        
        if len(df) < min_data_points:
            return [f"Insufficient data ({len(df)} {interval} points) for pattern recognition"]
        
        # Get recent data
        recent_data = df.tail(min_data_points)
        
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
        
        # Get last index
        last_idx = len(closes) - 1
        
        # Single candle patterns (need at least 1 recent candle)
        if last_idx >= 0:
            # Calculate candle metrics
            body = abs(closes[last_idx] - opens[last_idx])
            candle_range = highs[last_idx] - lows[last_idx]
            lower_shadow = min(opens[last_idx], closes[last_idx]) - lows[last_idx]
            upper_shadow = highs[last_idx] - max(opens[last_idx], closes[last_idx])
            
            # Hammer (long lower shadow, small upper shadow, bullish close)
            if (lower_shadow > body * shadow_ratio and
                upper_shadow < body * 0.1 and
                closes[last_idx] > opens[last_idx] and
                candle_range > 0):  # Ensure non-zero range
                patterns.append(f"Bullish Hammer ({interval})")
            
            # Hanging Man (similar to hammer but bearish close, in uptrend)
            if (lower_shadow > body * shadow_ratio and
                upper_shadow < body * 0.1 and
                closes[last_idx] < opens[last_idx] and
                last_idx >= trend_lookback and
                closes[last_idx-trend_lookback:last_idx].mean() < closes[last_idx]):  # In uptrend
                patterns.append(f"Bearish Hanging Man ({interval})")
            
            # Doji (very small body)
            if candle_range > 0 and body < candle_range * body_ratio:
                patterns.append(f"Doji Pattern ({interval})")
            
            # Shooting Star (long upper shadow, small lower shadow, bearish close)
            if (upper_shadow > body * shadow_ratio and
                lower_shadow < body * 0.1 and
                closes[last_idx] < opens[last_idx] and
                candle_range > 0):
                patterns.append(f"Bearish Shooting Star ({interval})")
            
            # Inverted Hammer (long upper shadow, small lower shadow, bullish close)
            if (upper_shadow > body * shadow_ratio and
                lower_shadow < body * 0.1 and
                closes[last_idx] > opens[last_idx] and
                candle_range > 0):
                patterns.append(f"Bullish Inverted Hammer ({interval})")
        
        # Two-candle patterns (need at least 2 candles)
        if last_idx >= 1:
            # Bullish Engulfing Pattern
            if (closes[last_idx-1] < opens[last_idx-1] and  # Previous is bearish
                closes[last_idx] > opens[last_idx] and      # Current is bullish
                opens[last_idx] < closes[last_idx-1] and    # Open below previous close
                closes[last_idx] > opens[last_idx-1]):      # Close above previous open
                patterns.append(f"Bullish Engulfing Pattern ({interval})")
            
            # Bearish Engulfing Pattern
            elif (closes[last_idx-1] > opens[last_idx-1] and  # Previous is bullish
                  closes[last_idx] < opens[last_idx] and      # Current is bearish
                  opens[last_idx] > closes[last_idx-1] and    # Open above previous close
                  closes[last_idx] < opens[last_idx-1]):      # Close below previous open
                patterns.append(f"Bearish Engulfing Pattern ({interval})")
            
            # Harami Pattern
            body_day1 = abs(closes[last_idx-1] - opens[last_idx-1])
            body_day2 = abs(closes[last_idx] - opens[last_idx])
            
            if body_day1 > body_day2 * engulfing_ratio:  # Day 1 has significantly larger body
                # Bullish Harami
                if (closes[last_idx-1] < opens[last_idx-1] and  # Day 1 bearish
                    closes[last_idx] > opens[last_idx] and      # Day 2 bullish
                    opens[last_idx] > closes[last_idx-1] and    # Day 2 open above Day 1 close
                    closes[last_idx] < opens[last_idx-1]):      # Day 2 close below Day 1 open
                    patterns.append(f"Bullish Harami Pattern ({interval})")
                # Bearish Harami
                elif (closes[last_idx-1] > opens[last_idx-1] and  # Day 1 bullish
                      closes[last_idx] < opens[last_idx] and      # Day 2 bearish
                      opens[last_idx] < closes[last_idx-1] and    # Day 2 open below Day 1 close
                      closes[last_idx] > opens[last_idx-1]):      # Day 2 close above Day 1 open
                    patterns.append(f"Bearish Harami Pattern ({interval})")
            
            # Piercing Pattern
            if (closes[last_idx-1] < opens[last_idx-1] and  # Day 1 bearish
                closes[last_idx] > opens[last_idx] and      # Day 2 bullish
                opens[last_idx] < closes[last_idx-1] and    # Day 2 open below Day 1 close
                closes[last_idx] > (opens[last_idx-1] + closes[last_idx-1]) / 2):  # Closes above midpoint
                patterns.append(f"Bullish Piercing Pattern ({interval})")
            
            # Dark Cloud Cover
            if (closes[last_idx-1] > opens[last_idx-1] and  # Day 1 bullish
                closes[last_idx] < opens[last_idx] and      # Day 2 bearish
                opens[last_idx] > closes[last_idx-1] and    # Day 2 open above Day 1 close
                closes[last_idx] < (opens[last_idx-1] + closes[last_idx-1]) / 2):  # Closes below midpoint
                patterns.append(f"Bearish Dark Cloud Cover ({interval})")
        
        # Three-candle patterns (need at least 3 candles)
        if last_idx >= 2:
            middle_candle_range = highs[last_idx-1] - lows[last_idx-1]
            
            # Morning Star (3-day bullish reversal)
            if (closes[last_idx-2] < opens[last_idx-2] and  # Day 1: bearish
                middle_candle_range > 0 and
                abs(closes[last_idx-1] - opens[last_idx-1]) < middle_candle_range * small_body_ratio and  # Day 2: small body
                closes[last_idx] > opens[last_idx] and  # Day 3: bullish
                closes[last_idx] > (opens[last_idx-2] + closes[last_idx-2]) / 2):  # Closes above midpoint of Day 1
                patterns.append(f"Bullish Morning Star ({interval})")
            
            # Evening Star (3-day bearish reversal)
            if (closes[last_idx-2] > opens[last_idx-2] and  # Day 1: bullish
                middle_candle_range > 0 and
                abs(closes[last_idx-1] - opens[last_idx-1]) < middle_candle_range * small_body_ratio and  # Day 2: small body
                closes[last_idx] < opens[last_idx] and  # Day 3: bearish
                closes[last_idx] < (opens[last_idx-2] + closes[last_idx-2]) / 2):  # Closes below midpoint of Day 1
                patterns.append(f"Bearish Evening Star ({interval})")
            
            # Three White Soldiers (3 consecutive bullish candles with higher closes)
            if all(closes[i] > opens[i] for i in range(last_idx-2, last_idx+1)):
                if (closes[last_idx-2] > opens[last_idx-2] and
                    closes[last_idx-1] > closes[last_idx-2] and
                    closes[last_idx] > closes[last_idx-1]):
                    patterns.append(f"Bullish Three White Soldiers ({interval})")
            
            # Three Black Crows (3 consecutive bearish candles with lower closes)
            if all(closes[i] < opens[i] for i in range(last_idx-2, last_idx+1)):
                if (closes[last_idx-2] < opens[last_idx-2] and
                    closes[last_idx-1] < closes[last_idx-2] and
                    closes[last_idx] < closes[last_idx-1]):
                    patterns.append(f"Bearish Three Black Crows ({interval})")
        
        # Add chart patterns with interval awareness
        patterns.extend(detect_chart_patterns(df, interval))
        
        return patterns if patterns else [f"No strong {interval} candlestick patterns detected"]
        
    except Exception as e:
        print(f"Pattern detection error for {interval}: {e}")
        return [f"Pattern recognition unavailable for {interval}"]

def detect_chart_patterns(df: pd.DataFrame, interval: str = 'daily') -> List[str]:
    """Detect chart patterns with interval awareness"""
    patterns = []
    
    try:
        min_points = 10 if interval == 'monthly' else 20
        
        if len(df) < min_points:
            return []
        
        closes = df['close'].values
        
        # Adjust parameters based on interval
        if interval == 'monthly':
            wedge_window = 12  # 12 months for wedge detection
            reversal_threshold = 0.03  # 3% for double top/bottom
        else:
            wedge_window = 20  # 20 days for wedge detection
            reversal_threshold = 0.02  # 2% for double top/bottom
        
        # Detect wedges
        if len(df) >= wedge_window:
            # Simple wedge detection (price converging)
            recent_closes = closes[-wedge_window:]
            high_trend = np.polyfit(range(wedge_window), 
                                   df['high'].values[-wedge_window:] if 'high' in df.columns else recent_closes, 
                                   1)[0]
            low_trend = np.polyfit(range(wedge_window), 
                                  df['low'].values[-wedge_window:] if 'low' in df.columns else recent_closes * 0.98, 
                                  1)[0]
            
            if high_trend < 0 and low_trend > 0:  # Converging
                if np.mean(recent_closes[-5:]) > np.mean(recent_closes[:5]):
                    patterns.append(f"Rising Wedge ({interval})")
                else:
                    patterns.append(f"Falling Wedge ({interval})")
        
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
            
            if len(peaks) >= 2:
                diff_pct = abs(peaks[-1] - peaks[-2]) / peaks[-1]
                if diff_pct < reversal_threshold:
                    if peaks[-1] < peaks[-2]:
                        patterns.append(f"Double Top ({interval})")
                    else:
                        patterns.append(f"Double Bottom Reversal ({interval})")
            
            if len(troughs) >= 2:
                diff_pct = abs(troughs[-1] - troughs[-2]) / troughs[-1]
                if diff_pct < reversal_threshold:
                    if troughs[-1] > troughs[-2]:
                        patterns.append(f"Double Bottom ({interval})")
                    else:
                        patterns.append(f"Double Top Reversal ({interval})")
        
        return patterns
        
    except Exception as e:
        print(f"Chart pattern detection error for {interval}: {e}")
        return []

def calculate_support_resistance(df: pd.DataFrame, interval: str = 'daily', current_price: Optional[float] = None) -> Dict:
    """Calculate support and resistance levels using FULL dataset, not downsampled"""
    
    # Interval-specific lookback
    if interval == 'monthly':
        lookback = min(24, len(df))  # Last 24 months
        min_required = 12
    else:  # daily
        lookback = min(90, len(df))  # Last 90 days
        min_required = 30
    
    if len(df) < min_required:
        return {
            'support': 0,
            'resistance': 0,
            'support_pivot': 0,
            'resistance_pivot': 0,
            'closest_level': f"Insufficient data ({len(df)} points)",
            'strength': "Weak"
        }
    
    try:
        # Use ORIGINAL full dataset, not downsampled
        recent = df.tail(lookback)
        high = float(recent['high'].max() if 'high' in recent.columns else recent['close'].max())
        low = float(recent['low'].min() if 'low' in recent.columns else recent['close'].min())
        close = float(recent['close'].iloc[-1])
        
        # Pivot points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        # Use provided current_price or fall back to last close
        price_to_use = current_price if current_price is not None else float(df['close'].iloc[-1])
        
        # Determine closest level
        levels = {
            'Resistance 2': r2,
            'Resistance 1': r1,
            'Pivot': pivot,
            'Support 1': s1,
            'Support 2': s2
        }
        
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - price_to_use))
        
        return {
            'support': round(s1, 2),
            'resistance': round(r1, 2),
            'support_2': round(s2, 2),
            'resistance_2': round(r2, 2),
            'pivot': round(pivot, 2),
            'closest_level': closest_level[0],
            'distance_pct': round(abs(closest_level[1] - price_to_use) / price_to_use * 100, 2),
            'strength': "Strong" if abs(closest_level[1] - price_to_use) / price_to_use < 0.02 else "Moderate",
            'lookback_period': f"{lookback} {interval} candles"
        }
        
    except Exception as e:
        print(f"Support/resistance calculation error: {e}")
        return {
            'support': 0,
            'resistance': 0,
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

def get_monthly_data(symbol: str) -> List[Dict]:
    """Get or calculate monthly data from daily data"""
    api_symbol = get_api_symbol(symbol)
    asset_ref = db.collection('historical_data').document(api_symbol)
    doc = asset_ref.get()
    
    if not doc.exists:
        return []
    
    data = doc.to_dict()
    daily_data = data.get('daily', [])
    
    if not daily_data:
        return []
    
    # Aggregate to monthly
    df = pd.DataFrame(daily_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Resample to monthly
    monthly_df = df.set_index('date').resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')
    
    return monthly_df.to_dict('records')

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
    
    # 1. DEFINE METADATA
    description = "Macro (4Y) Portfolio Strategy" if interval == 'monthly' else "Swing (6M) Technical Analysis"
    interval_type = "Monthly" if interval == 'monthly' else "Daily"
    
    try:
        # 2. Fetch from Firebase or sync
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        if asset_doc.exists:
            data_dict = asset_doc.to_dict()
        else:
            sync_result = sync_historical_data(symbol)
            if not sync_result.get('success'):
                return jsonify({"error": "Data sync failed", "message": sync_result.get('error')}), 503
            data_dict = db.collection('historical_data').document(api_symbol).get().to_dict()
        
        # 3. Get appropriate data
        data_pool = data_dict.get('monthly', []) if interval == 'monthly' else data_dict.get('daily', [])
        if not data_pool:
            data_pool = data_dict.get('daily', [])
            
        df = pd.DataFrame(data_pool)
        df['date'] = pd.to_datetime(df['date'])
        
        # 4. Apply timeframe filter
        if timeframe.endswith('y'):
            years_to_keep = int(timeframe.replace('y', ''))
            if interval == 'monthly':
                months_to_keep = years_to_keep * 12
                df = df.tail(months_to_keep)
            else:
                cutoff_date = datetime.now() - timedelta(days=years_to_keep * 365)
                df = df[df['date'] >= cutoff_date]
        elif timeframe.endswith('m'):
            months_to_keep = int(timeframe.replace('m', ''))
            if interval == 'monthly':
                df = df.tail(months_to_keep)
            else:
                days_to_keep = months_to_keep * 30
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                df = df[df['date'] >= cutoff_date]
        elif timeframe.endswith('d'):
            days_to_keep = int(timeframe.replace('d', ''))
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            df = df[df['date'] >= cutoff_date]

        # 5. **FETCH REAL-TIME PRICE** (NEW)
        realtime_price = get_realtime_price(symbol)
        historical_price = float(df['close'].iloc[-1])
        
        # Use real-time if available, otherwise fall back to historical
        current_price = realtime_price if realtime_price else historical_price
        price_source = "real-time" if realtime_price else "historical (last close)"
        
        # 6. Calculate MA on FULL timeframe
        ma_data = calculate_moving_averages(df)
        
        # 7. Get volume (last candle's total volume)
        latest_point = df.iloc[-1]
        current_volume = float(latest_point['volume'])
        
        # 8. DOWNSAMPLE to 50 points for AI context only
        analysis_df = downsample_to_50_points(df)
        
        # 9. **FIXED: Use FULL DF for Support/Resistance and Fibonacci**
        # Pass current_price to ensure real-time price is used in calculations
        fib_data = calculate_fibonacci_retracement(df, interval, current_price)
        sr_data = calculate_support_resistance(df, interval, current_price)
        patterns = detect_candlestick_patterns(analysis_df, interval) or ["No patterns detected"]
        
        # 10. Prepare technical summary
        technical_summary = {
            "symbol": symbol,
            "analysis_type": description,
            "current_price": current_price,
            "price_source": price_source,  # NEW
            "volume": current_volume,
            "price_formatted": format_currency(current_price),
            "data_period": f"{len(df)} {interval} candles (analyzed: {len(analysis_df)} points)",
            "data_source": data_dict.get('data_source', 'unknown'),
            "moving_averages": {
                "MA13": format_currency(ma_data['MA13']) if ma_data['MA13'] != "N/A" else "N/A",
                "MA20": format_currency(ma_data['MA20']) if ma_data['MA20'] != "N/A" else "N/A",
                "MA21": format_currency(ma_data['MA21']) if ma_data['MA21'] != "N/A" else "N/A",
                "MA50": format_currency(ma_data['MA50']) if ma_data['MA50'] != "N/A" else "N/A",
                "MA200": format_currency(ma_data['MA200']) if ma_data['MA200'] != "N/A" else "N/A",
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
                "current_vs_low": f"{fib_data.get('distance_from_low_pct', 0)}% above low",
                "lookback": fib_data.get('lookback_period', 'N/A')  # NEW
            },
            "support_resistance": {
                "support": format_currency(sr_data.get('support', 0)),
                "resistance": format_currency(sr_data.get('resistance', 0)),
                "pivot": format_currency(sr_data.get('pivot', 0)),
                "closest_level": sr_data.get('closest_level', 'Unknown'),
                "distance": f"{sr_data.get('distance_pct', 0)}%",
                "strength": sr_data.get('strength', 'Unknown'),
                "lookback": sr_data.get('lookback_period', 'N/A')  # NEW
            },
            "patterns_detected": patterns
        }
        
        # 11. Create AI prompt (updated with new info)
        prompt = f"""
        Perform a comprehensive technical analysis for {symbol} based on the following data:
        
        ANALYSIS TYPE: {description}
        CURRENT PRICE: {technical_summary['price_formatted']} ({price_source})
        DATA PERIOD: {technical_summary['data_period']}
        DATA SOURCE: {technical_summary['data_source'].upper()}
        
        MOVING AVERAGES:
        - MA13: {technical_summary['moving_averages']['MA13']}
        - MA20: {technical_summary['moving_averages']['MA20']}
        - MA21: {technical_summary['moving_averages']['MA21']}
        - MA50: {technical_summary['moving_averages']['MA50']}
        - MA200: {technical_summary['moving_averages']['MA200']}
        - Trend: {technical_summary['moving_averages']['trend']}
        - Alignment: {technical_summary['moving_averages']['alignment']}
        - Golden Cross: {technical_summary['moving_averages']['golden_cross']}
        
        FIBONACCI RETRACEMENT ({fib_data.get('lookback_period', 'N/A')}):
        - Current Zone: {technical_summary['fibonacci']['zone']}
        - Retracement Level: {technical_summary['fibonacci']['retracement_level']}
        - Description: {technical_summary['fibonacci']['description']}
        - Swing High: {technical_summary['fibonacci']['swing_high']}
        - Swing Low: {technical_summary['fibonacci']['swing_low']}
        - Position: {technical_summary['fibonacci']['current_vs_high']}, {technical_summary['fibonacci']['current_vs_low']}
        
        SUPPORT & RESISTANCE ({sr_data.get('lookback_period', 'N/A')}):
        - Support: {technical_summary['support_resistance']['support']}
        - Resistance: {technical_summary['support_resistance']['resistance']}
        - Pivot: {technical_summary['support_resistance']['pivot']}
        - Closest Level: {technical_summary['support_resistance']['closest_level']} ({technical_summary['support_resistance']['distance']} away)
        - Strength: {technical_summary['support_resistance']['strength']}
        
        PATTERNS DETECTED:
        {chr(10).join([f"- {pattern}" for pattern in patterns])}

        VOLUME: {technical_summary['volume']} ({interval_type} total)
        
        Based on ALL technical indicators above, provide a comprehensive analysis including:
        1. Overall trend assessment with confidence level
        2. Significance of Fibonacci retracement level (note the lookback period used)
        3. Moving average alignment implications (focus on 13, 20, 21, 50, 200 MA)
        4. Volume analysis: determine if participants are BUYING, SELLING, or MODERATE
        5. Volume confirmation of price trend
        6. Pattern recognition analysis
        7. Support/resistance breakout potential (note the lookback period used)
        8. Risk assessment with specific risk factors
        9. Clear trading recommendation with entry/exit levels
        
        Note: Current price is from {price_source}. If real-time, it may differ from the last historical close.
        
        Respond in valid JSON format with these exact fields:
        - trend: (string) Overall market trend with confidence
        - patterns: (array of strings) Key technical patterns with explanations
        - fibonacci_analysis: (string) Detailed Fibonacci interpretation
        - ma_analysis: (string) Moving average analysis with implications
        - support_resistance_analysis: (string) Breakout/breakdown potential
        - risk_assessment: (string) Specific risk factors (Low/Medium/High) - max 15 words
        - verdict: (string) Comprehensive analysis conclusion
        - suggestion: (string) BUY/HOLD/SELL with conviction
        - confidence_score: (number 0-100)
        - recommended_action: (string) Specific action with price targets
        - key_levels: (object) Important price levels to watch
        """

        # 12. Get AI response
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
        
        # 13. Combine all data
        result = {
            **ai_response,
            "technical_summary": technical_summary,
            "analysis_metadata": {
                "symbol": symbol,
                "interval": interval_type,
                "analysis_date": datetime.now().isoformat(),
                "data_points": len(df),
                "analyzed_points": len(analysis_df),
                "data_source": technical_summary['data_source'],
                "price_source": price_source
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
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        interval = data.get('interval', 'daily') # Capture the interval
        new_historical_data = data.get('data', [])
        
        if not symbol or not new_historical_data:
            return jsonify({"error": "Missing symbol or data"}), 400
        
        api_symbol = get_api_symbol(symbol)
        asset_ref = db.collection('historical_data').document(api_symbol)
        doc = asset_ref.get()
        
        # Determine which field to target (daily or monthly)
        target_field = 'monthly' if interval == 'monthly' else 'daily'
        
        existing_data = []
        full_doc_data = {}
        if doc.exists:
            full_doc_data = doc.to_dict()
            existing_data = full_doc_data.get(target_field, [])
        
        # Merge logic (preserving existing dates)
        existing_dates = {item['date'] for item in existing_data}
        merged_data = existing_data.copy()
        
        for new_item in new_historical_data:
            if new_item['date'] not in existing_dates:
                merged_data.append(new_item)
        
        merged_data.sort(key=lambda x: x['date'])
        
        # IMPORTANT: Update ONLY the target interval without deleting the other
        full_doc_data.update({
            target_field: merged_data,
            "symbol": symbol,
            "last_synced": datetime.now().isoformat(),
            "data_source": f"local_upload_{interval}"
        })
        
        asset_ref.set(full_doc_data, merge=False)
        
        return jsonify({"success": True, "message": f"Updated {target_field}"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from openai import OpenAI
import json

app = Flask(__name__)
CORS(app) # Mandatory for cross-origin requests from React

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- EXISTING CODE (KEEPING AS IS) ---
@app.route('/get-price', methods=['GET'])
def get_price():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    try:
        ticker = yf.Ticker(symbol)
        # fast_info is optimized for quick price retrieval
        price = ticker.fast_info['last_price']
        
        return jsonify({
            "symbol": symbol, 
            "price": price
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    symbol = request.args.get('symbol')
    # Default period is 1 month; other options: 1d, 5d, 6mo, 1y, etc.
    period = request.args.get('period', '1mo') 
    
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    try:
        ticker = yf.Ticker(symbol)
        # Fetch OHLC data for the requested period
        df = ticker.history(period=period)
        
        if df.empty:
            return jsonify({"error": "No historical data found for this symbol"}), 404

        # Format the data for the AI prompt and frontend charts
        historical_list = []
        for date, row in df.iterrows():
            historical_list.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
            
        return jsonify({
            "symbol": symbol,
            "period": period,
            "data": historical_list
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1mo')
    interval = request.args.get('interval', '1d')
    
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    # 1. Improved symbol handling for yfinance
    api_symbol = symbol
    if symbol in ['BTC', 'ETH', 'SOL'] and not symbol.endswith('-USD'):
        api_symbol = f"{symbol}-USD"
    elif not symbol.endswith('.JK') and any(s in symbol for s in ['BBCA', 'TLKM', 'ASII']):
        api_symbol = f"{symbol}.JK"

    try:
        ticker = yf.Ticker(api_symbol)
        # 2. Add a timeout to history fetch to prevent hanging
        df = ticker.history(period=timeframe, interval=interval)
        
        if df.empty:
            return jsonify({"error": f"Yahoo Finance returned no data for {api_symbol}. Try a different timeframe."}), 404
        
        # 3. Use standard columns (yfinance columns are Title Case)
        # Check if columns exist before using them to avoid KeyError
        cols = [c for c in ['Close', 'High', 'Low', 'Volume'] if c in df.columns]
        data_summary = df.tail(30).to_csv(columns=cols)

        # 4. OpenAI Call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior financial analyst. Respond ONLY in valid JSON format." 
                },
                {
                    "role": "user", 
                    "content": f"Analyze this OHLC data for {api_symbol} and return a JSON object with keys 'trend', 'patterns', 'sentiment_score', 'verdict', and 'suggestion':\n{data_summary}"
                }
            ],
            response_format={ "type": "json_object" }
        )
        
        ai_data = json.loads(response.choices[0].message.content)
        return jsonify(ai_data)

    except Exception as e:
        # This will print the EXACT error (e.g., 'insufficient_quota') in Render logs
        print(f"CRITICAL ERROR: {str(e)}") 
        return jsonify({"error": f"AI Engine Error: {str(e)}"}), 500 

if __name__ == '__main__':
    app.run(port=5000)


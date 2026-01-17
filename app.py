import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from openai import OpenAI

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

# --- NEW OHLC DATA ENDPOINT ---
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
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    try:
        # 1. Fetch 30 days of historical data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo")
        
        # 2. Format data into a compact string to save tokens
        # Date, Close, High, Low
        data_summary = df.tail(30).to_csv(columns=['Close', 'High', 'Low'])

        # 3. Call OpenAI (using gpt-4o-mini for speed and cost)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst specializing in technical analysis and pattern recognition."},
                {"role": "user", "content": f"Analyze the following 30-day OHLC data for {symbol}:\n{data_summary}\n\nTasks:\n1. Identify trend (Bullish/Bearish/Sideways).\n2. Spot patterns (Support/Resistance, Breakouts).\n3. Give a Sentiment Score (1-100).\n4. Provide a 2-sentence 'Executive Verdict'."}
            ],
            response_format={ "type": "json_object" } # Ensures we get clean JSON back
        )
        
        # Parse the AI response
        ai_analysis = response.choices[0].message.content
        return ai_analysis # This will return a JSON object to your React app

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)


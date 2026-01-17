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

app = Flask(__name__)
CORS(app)

# 1. INITIALIZATION
# Ensure serviceAccountKey.json is in your backend folder
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_api_symbol(symbol):
    """Adds -USD for crypto and .JK for Indo stocks."""
    if symbol in ['BTC', 'ETH', 'SOL', 'BNB'] and not symbol.endswith('-USD'):
        return f"{symbol}-USD"
    if any(s in symbol for s in ['BBCA', 'TLKM', 'ASII']) and not symbol.endswith('.JK'):
        return f"{symbol}.JK"
    return symbol

# 2. ENDPOINTS
@app.route('/get-price', methods=['GET'])
def get_price():
    symbol = request.args.get('symbol')
    if not symbol: return jsonify({"error": "Symbol required"}), 400
    try:
        ticker = yf.Ticker(get_api_symbol(symbol))
        return jsonify({"symbol": symbol, "price": ticker.fast_info['last_price']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    """Syncs 10 years of data to Firebase (Daily & Monthly arrays)."""
    symbol = request.args.get('symbol')
    api_symbol = get_api_symbol(symbol)
    asset_ref = db.collection('historical_data').document(api_symbol)
    doc = asset_ref.get()

    # Default to fetching everything since 2013 if no doc exists
    start_sync = "2013-01-01"
    existing_daily = []
    existing_monthly = []

    if doc.exists:
        data = doc.to_dict()
        existing_daily = data.get('daily', [])
        existing_monthly = data.get('monthly', [])
        if existing_daily:
            start_sync = (datetime.strptime(existing_daily[-1]['date'], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    # SYNC NEW DATA
    try:
        # Fetch Daily
        new_df_d = yf.download(api_symbol, start=start_sync, interval="1d")
        # Fetch Monthly
        new_df_m = yf.download(api_symbol, start=start_sync, interval="1mo")

        def df_to_list(df):
            return [{"date": d.strftime('%Y-%m-%d'), "close": round(float(r['Close']), 2)} 
                    for d, r in df.iterrows()] if not df.empty else []

        updated_daily = existing_daily + df_to_list(new_df_d)
        updated_monthly = existing_monthly + df_to_list(new_df_m)

        asset_ref.set({
            "daily": updated_daily,
            "monthly": updated_monthly,
            "last_updated": datetime.now().isoformat()
        }, merge=True)

        return jsonify({"daily": updated_daily, "monthly": updated_monthly})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-insight', methods=['GET'])
def get_ai_insight():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval', 'daily') # 'daily' or 'monthly'
    api_symbol = get_api_symbol(symbol)

    try:
        # 1. Get Data from Firebase
        asset_doc = db.collection('historical_data').document(api_symbol).get()
        if not asset_doc.exists: return jsonify({"error": "Sync data first"}), 404
        
        full_data = asset_doc.to_dict().get(interval, [])
        df = pd.DataFrame(full_data)
        
        # 2. Apply Windows & Indicators
        if interval == 'daily':
            # 6 Months Window (approx 126 trading days)
            df_window = df.tail(126).copy()
            # Sample to 30 points for AI
            df_sampled = df_window.iloc[::4].tail(30) 
        else:
            # 48 Months Window
            df_window = df.tail(48).copy()
            df_sampled = df_window # Send all 48 points for monthly macro

        # 3. Calculate indicators to "Teach" the AI
        df_window['MA20'] = df_window['close'].rolling(window=20).mean()
        curr_ma = round(df_window['MA20'].iloc[-1], 2)
        
        # 4. Fetch News
        news = [n['title'] for n in yf.Ticker(api_symbol).news[:5]]

        # 5. Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior analyst. Respond ONLY in valid JSON. No nested objects."},
                {"role": "user", "content": f"Analyze {api_symbol} ({interval}).\nData:\n{df_sampled.to_csv()}\nIndicators: MA20 is {curr_ma}.\nNews: {news}"}
            ],
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
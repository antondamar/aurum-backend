# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)
CORS(app) # Mandatory for cross-origin requests from React

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

if __name__ == '__main__':
    app.run(port=5000)
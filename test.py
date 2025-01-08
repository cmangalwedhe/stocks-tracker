from flask import Flask, render_template, request, jsonify
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

app = Flask(__name__)

# Store stocks list in memory
STOCKS = ['agx', 'alab', 'arm', 'asml', 'aspn', 'cava', 'crwd', 'deck', 'dell', 'dkng',
          'duol', 'elf', 'estc', 'four', 'glbe', 'hims', 'hood', 'klac', 'mdb', 'meli',
          'meta', 'mndy', 'mu', 'net', 'nu', 'onon', 'rbrk', 'rddt', 'rklb', 'rxrx',
          'smcl', 'snow', 'tdw', 'tmdx', 'tsla', 'u', 'uber', 'vktx', 'zs']

# Cache duration in seconds (e.g., 5 minutes)
CACHE_DURATION = 300


def get_cached_timestamp():
    return time.time() // CACHE_DURATION


@lru_cache(maxsize=1)
def get_all_stock_data(timestamp):
    """Fetch all stock data in parallel with caching"""

    def fetch_single_stock(symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.get_info()
            price_target = ticker.analyst_price_targets.get('mean', 'N/A')

            return {
                'symbol': symbol.upper(),
                'longName': info.get('longName', 'N/A'),
                'currentPrice': info.get('currentPrice', 'N/A'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
                'fiftyDayAverage': info.get('fiftyDayAverage', 'N/A'),
                'twoHundredDayAverage': info.get('twoHundredDayAverage', 'N/A'),
                'oneYearTargetEstimate': price_target,
                'url': f'https://finance.yahoo.com/quote/{symbol}/'
            }
        except:
            return None

    # Use ThreadPoolExecutor to fetch data in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_stock, STOCKS))

    return [r for r in results if r is not None]


@app.route('/')
def index():
    # Get current cache timestamp
    current_timestamp = get_cached_timestamp()

    # Get cached or fetch new stock data
    stock_data = get_all_stock_data(current_timestamp)
    return render_template('index.html', stocks=stock_data)


@app.route('/add_stock', methods=['POST'])
def add_stock():
    symbol = request.form.get('symbol', '').strip().lower()

    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    if symbol in STOCKS:
        return jsonify({'success': False, 'message': 'Stock already in list'})

    # Verify the stock exists
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
        if info:
            STOCKS.append(symbol)
            STOCKS.sort()
            # Invalidate cache by updating the timestamp
            get_all_stock_data.cache_clear()
            return jsonify({'success': True, 'message': 'Stock added successfully'})
    except:
        pass

    return jsonify({'success': False, 'message': 'Invalid stock symbol'})


@app.route('/delete_stock', methods=['POST'])
def delete_stock():
    symbol = request.form.get('symbol', '').strip().lower()

    if symbol in STOCKS:
        STOCKS.remove(symbol)
        # Invalidate cache
        get_all_stock_data.cache_clear()
        return jsonify({'success': True, 'message': 'Stock removed successfully'})

    return jsonify({'success': False, 'message': 'Stock not found'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
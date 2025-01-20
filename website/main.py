from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import time
from functools import lru_cache
from models import User

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
    stocks = current_user.get_stock_symbols()
    print(stocks)

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
        results = list(executor.map(fetch_single_stock, stocks))

    return [r for r in results if r is not None]


main = Blueprint('main', __name__)


@main.route('/add_stock', methods=['POST'])
@login_required
def add_stock():
    symbol = request.form.get('symbol', '').strip().lower()
    stocks = current_user.get_stock_symbols()

    if not symbol:
        return jsonify({'success': False, 'message': 'No symbol provided'})

    if symbol in stocks:
        return jsonify({'success': False, 'message': 'Stock already in list'})

    # Verify the stock exists
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
        if info:
            current_user.add_stock(symbol)
            # STOCKS.sort()
            # Invalidate cache by updating the timestamp
            get_all_stock_data.cache_clear()
            return jsonify({'success': True, 'message': 'Stock added successfully'})
    except:
        pass

    return jsonify({'success': False, 'message': 'Invalid stock symbol'})


@main.route('/delete_stock', methods=['POST'])
@login_required
def delete_stock():
    symbol = request.form.get('symbol', '').strip().lower()
    stocks = current_user.get_stock_symbols()

    if symbol in stocks:
        # STOCKS.remove(symbol)
        current_user.remove_stock(symbol)
        # Invalidate cache
        get_all_stock_data.cache_clear()
        return jsonify({'success': True, 'message': 'Stock removed successfully'})

    return jsonify({'success': False, 'message': 'Stock not found'})


@main.route('/')
def index():
    return current_user.get_first_name()


@main.route('/profile')
@login_required
def profile():
    timestamp = get_cached_timestamp()
    get_all_stock_data.cache_clear()

    stock_data = get_all_stock_data(timestamp)
    return render_template('index.html', stocks=stock_data)

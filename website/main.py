from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import time
from functools import lru_cache
from models import User
import lstm_model
from datetime import datetime as dt


# Cache duration in seconds (e.g., 5 minutes)
CACHE_DURATION = 300


def get_cached_timestamp():
    return time.time() // CACHE_DURATION


@lru_cache(maxsize=1)
def get_all_stock_data(timestamp):
    stocks = current_user.get_stock_symbols()

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
        except Exception as e:
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
        current_user.remove_stock(symbol)
        # Invalidate cache
        get_all_stock_data.cache_clear()
        return jsonify({'success': True, 'message': 'Stock removed successfully'})

    return jsonify({'success': False, 'message': 'Stock not found'})


@main.route('/')
def index():
    return render_template('profile.html')


@main.route('/prediction/<ticker>')
def prediction(ticker):
    plot_and_predictions = lstm_model.predict_stock_prices(ticker, days_to_predict=5)
    plot = plot_and_predictions[1]
    predictions = plot_and_predictions[0]

    print(predictions)

    return render_template("prediction.html", ticker_symbol = ticker,
                                                                current_date = dt.now(),
                                                                days=[i for i in range(1, len(predictions)+1)],
                                                                chart_image=plot,
                                                                prediction=predictions)




@main.route('/tos')
def tos():
    return render_template("tos.html")


@main.route('/privacy-policy')
def privacy_policy():
    return render_template("privacy_policy.html")


@main.route('/contact-us')
def contact_us():
    return render_template("contact_us.html")

@main.route('/profile')
@login_required
def profile():
    timestamp = get_cached_timestamp()
    get_all_stock_data.cache_clear()
    stock_data = get_all_stock_data(timestamp)

    print(stock_data)

    return render_template('index.html', stocks=stock_data)

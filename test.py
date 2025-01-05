# app.py
from flask import Flask, render_template
import yfinance as yf

app = Flask(__name__)


@app.route('/')
def index():
    stonks = ['agx', 'alab', 'arm', 'asml', 'aspn', 'cava', 'crwd', 'deck', 'dell', 'dkng',
              'duol', 'elf', 'estc', 'four', 'glbe', 'hims', 'hood', 'klac', 'mdb', 'meli',
              'meta', 'mndy', 'mu', 'net', 'nu', 'onon', 'rbrk', 'rddt', 'rklb', 'rxrx',
              'smcl', 'snow', 'tdw', 'tmdx', 'tsla', 'u', 'uber', 'vktx', 'zs']
    stock_data = []

    for stonk in stonks:
        info = yf.Ticker(stonk).get_info()

        stock_info = {
            'symbol': stonk.upper(),
            'longName': info.get('longName', 'N/A'),
            'currentPrice': info.get('currentPrice', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fiftyDayAverage': info.get('fiftyDayAverage', 'N/A'),
            'twoHundredDayAverage': info.get('twoHundredDayAverage', 'N/A'),
            'oneYearTargetEstimate': yf.Ticker(stonk).analyst_price_targets.get('mean', 'N/A'),
            'url': f'https://finance.yahoo.com/quote/{stonk}/'
        }
        stock_data.append(stock_info)

    return render_template('index.html', stocks=stock_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
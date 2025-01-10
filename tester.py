import yfinance as yf

tick = yf.Ticker('rbrk')
print(tick.get_info())
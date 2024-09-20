import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y") 
    return hist

if __name__ == "__main__":
    data = get_stock_data("AAPL")
    print(data.tail())  
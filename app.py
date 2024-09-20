from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_last_step = lstm_out[:, -1, :]  
        predictions = self.linear(lstm_last_step)
        return predictions


model = LSTM()
model.load_state_dict(torch.load('stock_lstm_model.pth'))  
model.eval()  

scaler = pd.read_pickle('scaler.pkl')  


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist


def prepare_data(data, time_step=60):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
    return np.array(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    hist = get_stock_data(ticker)
    dates = hist.index  
    close_prices = hist['Close'].values  

    
    data_scaled = scaler.transform(np.array(close_prices).reshape(-1, 1))

    time_step = 60
    X_input = prepare_data(data_scaled, time_step)

    
    X_input = torch.FloatTensor(X_input).view(-1, time_step, 1)

    
    future_days = 5
    predictions = []
    input_seq = X_input[-1].unsqueeze(0) 

    with torch.no_grad():
        for _ in range(future_days):
            
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            
            
            y_pred = model(input_seq)  
            predictions.append(y_pred.item())
            
            
            y_pred = y_pred.unsqueeze(-1)  

            
            new_input = torch.cat((input_seq[:, 1:, :], y_pred), dim=1) 
            input_seq = new_input

    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d')

    
    current_price = close_prices[-1]
    predicted_price = predictions[0][0]
    trace_actual = go.Scatter(x=dates, y=close_prices, mode='lines', name='Actual Price')
    trace_predicted = go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines+markers', name='Predicted Price', line=dict(dash='dash', color='red'))

    layout = go.Layout(
        title='Price History and Prediction',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        hovermode='closest'
    )
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    graph_div = pio.to_html(fig, full_html=False)

    return render_template('result.html', current_price=current_price, predicted_price=predicted_price, graph_div=graph_div)

if __name__ == "__main__":
    app.run(debug=True)
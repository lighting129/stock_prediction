import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  
    return hist['Close'].values  


def prepare_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_model(data, num_epochs=10, time_step=60):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = prepare_data(data_scaled, time_step)

    
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)

    
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(num_epochs):
        for i in range(len(X_train)):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(X_train[i])

            single_loss = loss_function(y_pred, y_train[i].view(-1))
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {single_loss.item()}')

    
    torch.save(model.state_dict(), 'stock_lstm_model.pth')
    pd.to_pickle(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    
    stock_data = get_stock_data('AAPL')
    train_model(stock_data)
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# Get stock symbol from user input
stock_symbol = input("Enter the stock symbol (e.g., 'IRFC.NS'): ")

# Fetch stock data
stock = yf.Ticker(stock_symbol)
data = stock.history(period="1y")  # Last one year of data

# Check if data is empty
if data.empty:
    raise ValueError("No data found for the given stock symbol. Please check the symbol and try again.")

# Prepare data
data = data[['Close']]
data = data.reset_index()

# Convert dates to numeric format
data['Date'] = pd.to_datetime(data['Date'])
data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Prepare sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train model
history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.1, callbacks=callbacks)

# Load the best model
model.load_weights('best_model.keras')

# Predict
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test_inv = scaler.inverse_transform(y_test)

# Evaluate model
mse = mean_squared_error(y_test_inv, predicted_prices)
print(f"Mean Squared Error: {mse:.2f}")

# Forecast next working days
last_sequence = scaled_data[-seq_length:]
forecast = []

for _ in range(10):  # Forecast for 10 days
    next_seq = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(next_seq)
    forecast.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Generate forecast dates (account for weekends)
forecast_dates = []
current_date = data['Date'].max() + pd.DateOffset(days=1)

while len(forecast_dates) < len(forecast):
    if current_date.weekday() < 5:  # Check if it's a weekday
        forecast_dates.append(current_date)
    current_date += pd.DateOffset(days=1)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Price (INR)': forecast.flatten()
})

# Determine color for the forecast line based on price change
if forecast_df['Forecasted Price (INR)'].iloc[-1] > data['Close'].iloc[-1]:
    line_color = 'green'
else:
    line_color = 'red'

# Plot results
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Price'))

# Forecast data
fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Price (INR)'],
                         mode='lines', name='Forecasted Price', line=dict(dash='dash', color=line_color)))

fig.update_layout(title=f'Stock Price Forecast for {stock_symbol}',
                  xaxis_title='Date',
                  yaxis_title='Price (INR)',
                  template='plotly_dark')

# Display results
fig.show()

# Display prediction results
current_price = data['Close'].iloc[-1]
next_day_price = forecast_df['Forecasted Price (INR)'].iloc[0]

percentage_change = ((next_day_price - current_price) / current_price) * 100

# Long-term trend analysis
long_term_trend = np.mean(np.diff(forecast.flatten()))

# Calculate standard deviation of prediction error
prediction_errors = y_test_inv - predicted_prices
prediction_error_std = np.std(prediction_errors)

# Calculate confidence interval for next day's price
z_score = norm.ppf(0.975)  # 95% confidence interval
confidence_interval = z_score * prediction_error_std

# Calculate recommendation percentage
if percentage_change > confidence_interval:
    recommendation = "Buy"
    buy_confidence = min(percentage_change / confidence_interval * 100, 100)
    sell_confidence = 0
    hold_confidence = 100 - buy_confidence
elif percentage_change < -confidence_interval:
    recommendation = "Sell"
    sell_confidence = min(-percentage_change / confidence_interval * 100, 100)
    buy_confidence = 0
    hold_confidence = 100 - sell_confidence
else:
    recommendation = "Hold"
    buy_confidence = 0
    sell_confidence = 0
    hold_confidence = 100

print(f"Current Price: ₹{current_price:.2f}")
print(f"Forecasted Price for Next Working Day: ₹{next_day_price:.2f}")
print(f"Percentage Change: {percentage_change:.2f}%")
print(f"Recommendation: {recommendation}")
print(f"Buy Confidence: {buy_confidence:.2f}%")
print(f"Sell Confidence: {sell_confidence:.2f}%")
print(f"Hold Confidence: {hold_confidence:.2f}%")

# Save recommendation to forecast_df
forecast_df['Recommendation'] = recommendation
forecast_df['Buy Confidence (%)'] = buy_confidence
forecast_df['Sell Confidence (%)'] = sell_confidence
forecast_df['Hold Confidence (%)'] = hold_confidence
print(forecast_df)

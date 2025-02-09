# Stock Price Prediction and Recommendation System

## Overview

This project implements a stock price prediction and recommendation system using a bidirectional Long Short-Term Memory (LSTM) model. It provides historical stock prices, forecasts future prices, and gives recommendations (Buy, Sell, Hold) with confidence percentages.

<div>
  <img src="assets/image.png" style="width:600px;height:500px" alt="Stock Price Prediction Example">
    <img src="assets/result1.png" style="width:600px;height:500px" alt="Stock Price Prediction Example">
    <img src="assets/result2.png" style="width:450px;height:300px" alt="Stock Price Prediction Example">
</div>

## Features

- **Historical Data Visualization**: Fetch and display the last year's stock prices.
- **Future Price Prediction**: Predict the next 10 working days' stock prices using a bidirectional LSTM model.
- **Trend Analysis**: Analyze long-term trends to inform recommendations.
- **Recommendation System**: Provides Buy, Sell, or Hold recommendations with confidence percentages.
- **Dynamic Visualization**: Interactive plot displaying historical prices and forecasted prices.

## Requirements

- Python 3.x
- pandas
- numpy
- yfinance
- plotly
- keras
- scikit-learn
- scipy

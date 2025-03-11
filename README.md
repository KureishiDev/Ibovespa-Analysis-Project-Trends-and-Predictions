# Stock Price Prediction with Ibovespa Data

This project demonstrates a step-by-step approach to analyzing and forecasting the Ibovespa (Brazilian Stock Exchange Index) prices using various techniques, including ARIMA, Prophet, and Linear Regression. We will download historical data, perform exploratory data analysis, apply time series models, and evaluate prediction performance.

## Table of Contents
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Time Series Analysis](#time-series-analysis)
- [Feature Engineering](#feature-engineering)
- [Predictive Modeling](#predictive-modeling)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Installation
Ensure you have Python installed (preferably version 3.8 or above). Install the required packages with:

```bash
pip install yfinance pandas matplotlib seaborn scikit-learn statsmodels prophet
```

## Data Collection
We collect historical data of the Ibovespa index from Yahoo Finance using `yfinance`.

```python
ibov_data = yf.download('^BVSP', start='2020-01-01', end='2023-01-01')
ibov_data.dropna(inplace=True)
```

## Exploratory Data Analysis (EDA)
We visualize the Adjusted Close Price and add moving averages to observe trends.

- Plot of Adjusted Close Price
- Plot with Moving Averages (50-day and 200-day)

## Time Series Analysis
Two forecasting models are used:

1. **ARIMA Model:**
```python
model_arima = ARIMA(ibov_data['Adj Close'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=30)
```

2. **Prophet Model:**
```python
df_prophet = ibov_data['Adj Close'].reset_index()
df_prophet.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)
```

## Feature Engineering
We add lag features and moving averages to enhance the predictive modeling.
```python
ibov_data['Lag_1'] = ibov_data['Adj Close'].shift(1)
ibov_data.dropna(inplace=True)
```

## Predictive Modeling
We apply Linear Regression to predict the Adjusted Close price.
```python
features = ['Lag_1', 'MA_50', 'MA_200']
X_train, X_test, y_train, y_test = train_test_split(ibov_data[features], ibov_data['Adj Close'], test_size=0.2, random_state=42)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
```

## Results
The models are evaluated using Mean Squared Error (MSE) and compared visually.

## Future Improvements
- Adding more features to improve the linear regression model.
- Applying more complex models (e.g., LSTM, XGBoost).
- Enhancing feature engineering techniques.

## License
This project is licensed under the MIT License.


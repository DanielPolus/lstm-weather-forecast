# LSTM Weather Forecast + Agronomy Tips

This project predicts short-term weather metrics (temperature, humidity, precipitation, wind) using an **LSTM** model on historical data from **Meteostat**, and provides simple sowing recommendations based on forecasted conditions.  
It includes a **Tkinter GUI** to choose a city, fetch data, visualize historical trends, generate forecasts, and display crop recommendations.

## Key Features
- Fetches **hourly/daily** weather data from [Meteostat](https://meteostat.net/).
- Decomposition + normalization of time series (trend + seasonality via `statsmodels.seasonal_decompose`).
- **LSTM (PyTorch)** model for next-day / short horizon forecasting (sliding window).
- **Caching**: CSV cache per city in `data_cache/` to avoid repeated downloads.
- **Tkinter GUI** with Matplotlib plots (4 subplots: humidity, temperature, precipitation, wind).
- Simple **agronomy rules** to recommend crops based on forecast ranges.

## How It Works (Pipeline)
1. **Data loading**: from Meteostat (hourly/daily). On first run, data is downloaded and cached into `data_cache/<city>_*.csv`.  
   On subsequent runs, the app asks whether to reuse cached data.
2. **Preprocessing**: interpolate/bfill → seasonal decomposition → scale trend to [0,1].
3. **Supervised dataset**: rolling window sequences (e.g., 30 days → predict next).
4. **Model**: simple **LSTM → Linear** (MSE loss, Adam).
5. **Forecast**: iterative, de-normalize back, add seasonality tail.
6. **GUI**: plots forecasts and shows basic **sowing recommendations** for a set of crops (threshold-based rules).

## Install
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

python main.py
```

On first data access you’ll be asked whether to use cached data or download fresh data.

Usage
- Choose a city (combobox).
- Click Generate Forecast → trains LSTM on last ~2 years (configurable) and predicts the next 14 days.
- Click Show Historical Data → plots last 30 days of observed metrics.
- Click Get Recommendations → shows simple crop suggestions based on forecasted avg conditions.

Configuration
- Sequence length (seq_length), forecast horizon (forecast_days), LSTM layers/hidden size, learning rate, and epochs are set in code (easy to tweak).
- Crops and their optimal ranges are in show_recommendations(); adjust thresholds to your needs.

Known Limitations
- Simple LSTM; no exogenous vars, no cross-validation, no hyperparameter search.
- Recommendations are rule-based (heuristics), not ML-based agronomy.
- Seasonal decomposition assumes sufficient history and stationarity of patterns.
- Meteostat availability depends on station coverage for your location.

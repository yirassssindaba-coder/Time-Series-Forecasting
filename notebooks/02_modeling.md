# Notebook: 02 - Modelling (Panduan)
- Contoh penggunaan:
  - Prophet: ubah dataframe jadi ['ds','y'] dan fit (src/models/prophet_model.py)
  - LightGBM: gunakan fitur lag/rolling (src/features/feature_engineering.py)
  - LSTM: buat window sliding, ubah skala, reshape ke (samples, timesteps, features)
- Evaluasi: MAE, RMSE, MAPE, dan backtesting rolling-origin
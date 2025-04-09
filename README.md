# Web Traffic Time‑Series Forecasting with an Ensemble of ARIMA + LSTM

This repository contains the code, data‑processing pipelines, and experiment notebooks for my master’s capstone project: **forecasting Wikipedia page‑view traffic with a hybrid statistical + deep‑learning model**. The core idea is simple yet powerful—combine the interpretability of classical time‑series methods with the pattern‑recognition muscle of recurrent neural networks to deliver forecasts that are both accurate and robust.

## Why this project?
Accurately predicting traffic spikes is mission‑critical for any content or e‑commerce platform. Even a few hundred milliseconds of extra latency can translate into lost ad revenue or abandoned carts. By modeling 18 months of daily page‑view counts across English Wikipedia, the project demonstrates how an ensemble can outperform either model in isolation, providing a practical blueprint for capacity planning, marketing timing, and content scheduling.

## Data set
The raw data—released for a Kaggle competition by Google—contains daily views for ~145 k Wikipedia articles (July 2015 – Dec 2016). For this study, I aggregated **all English‑language pages** into a single univariate series, filled true missing days with zeros, and resampled to enforce a daily frequency. Exploratory analysis confirmed clear seasonality plus occasional exogenous spikes (e.g., viral news events) citeturn0file0.

## Modeling approach
1. **Seasonal ARIMA (SARIMA)** via *pmdarima*’s auto‑grid‑search captures linear trends and seasonality.
2. **LSTM network** in TensorFlow handles nonlinear, long‑range dependencies; data is windowed with `TimeseriesGenerator` and scaled for stability.
3. **Ensemble strategy**: final predictions are the simple mean of the two model outputs, leveraging the complementary error profiles.

## Results
| Model | RMSE | SMAPE |
|-------|------|-------|
| SARIMA | 13.34 M | 8 % |
| LSTM   | 13.77 M | 8 % |
| **Ensemble** | **11.52 M** | **6 %** |

The hybrid cut error by **≈14 % (RMSE)** and improved directional accuracy, validating the hypothesis that mixing linear and nonlinear perspectives yields superior forecasts.

## Future work
Next steps include incorporating exogenous signals (social‑media trends, search‑algorithm updates) and experimenting with lightweight Transformer variants to further reduce error without sacrificing interpretability.

Feel free to open issues or PRs—feedback and collaboration are welcome!

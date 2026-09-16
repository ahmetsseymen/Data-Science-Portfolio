# Flight Delay Prediction

A machine learning project that investigates whether flight arrival delay can be predicted more effectively **directly** or by decomposing it into operational components.

The project uses historical U.S. domestic flight data together with origin and destination weather information available during the **final hour before scheduled departure**.

## Project Status

Completed

## Objective

The main research question of the project is:

> Is it more effective to predict arrival delay directly, or to model its operational components separately and combine their predictions?

Arrival delay can be represented as:

`ArrDelay = DepDelay + Elapsed_Time_Deviation`

where:

- `DepDelay` represents departure schedule deviation.
- `Elapsed_Time_Deviation` represents the difference between actual and scheduled gate-to-gate flight duration.
- `ArrDelay` represents arrival schedule deviation.

Three regression models were therefore constructed:

- **M1:** Predict `DepDelay`
- **M2:** Predict `Elapsed_Time_Deviation`
- **M3:** Predict `ArrDelay` directly

The component prediction is calculated as:

`Component ArrDelay = M1 + M2`

An additional ensemble combines the component and direct predictions:

`Final Ensemble = 0.75 × Component + 0.25 × Direct`

## Prediction Scenario

The prediction point falls within the **final hour before scheduled departure (approximately 1–59 minutes before departure)**.

Only information intended to be available at the prediction time is used as model input.

Post-flight variables such as actual elapsed time, taxi times, delay reason variables, and realized arrival information are excluded from model features.

## Data Source

Flight data:

- U.S. Department of Transportation
- Bureau of Transportation Statistics (BTS)
- Reporting Carrier On-Time Performance Data

Historical period:

- **Development period:** July 2025 - May 2026
- **Final unseen test:** June 2026

Weather data was integrated for both origin and destination airports at the same physical prediction time within the final pre-departure hour.

## Data Preparation

Major preprocessing steps include:

- Removal of cancelled and diverted flights
- Data quality and consistency checks
- Validation of the relationship between departure, elapsed-time, and arrival deviations
- Airport code standardization
- Time-based train and validation split
- Frequency Encoding for high-cardinality variables such as:
  - `Origin`
  - `Dest`
  - `Route`
- One-Hot Encoding for remaining categorical variables
- Standardization of continuous numerical features
- Prevention of validation and final-test leakage by fitting preprocessing operations only on training data

## Feature Engineering

Examples of engineered features include:

- Directional flight route
- Season
- Weekend indicator
- Scheduled departure hour
- Scheduled arrival hour
- Departure time period
- Arrival time period
- Planned average gate-to-gate speed
- Origin airport latitude and longitude
- Pre-departure prediction timestamp

Weather features were added for both origin and destination airports:

- Temperature
- Precipitation
- Snowfall
- Weather condition code
- Mean sea-level pressure
- Cloud cover
- Visibility
- Wind speed
- Wind direction
- Wind gusts

## Models Evaluated

Baseline regression algorithms included:

- Ridge Regression
- Extra Trees Regressor
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoostingRegressor

Based on validation performance, the selected models were:

- **M1 - DepDelay:** HistGradientBoostingRegressor
- **M2 - Elapsed Time Deviation:** CatBoostRegressor
- **M3 - ArrDelay:** XGBoostRegressor

CatBoost hyperparameters for M2 were further manually optimized.

## Evaluation Metrics

Regression models were evaluated using:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **R² — Coefficient of Determination**

Lower MAE and RMSE values indicate better performance, while higher R² values are preferred.

## Final Test Results

The final models were evaluated on previously unseen **June 2026** flight data.

| Approach | MAE | RMSE | R² |
|---|---:|---:|---:|
| Direct (M3) | 29.69 | 62.83 | 0.0508 |
| Component (M1 + M2) | **29.23** | 62.39 | 0.0640 |
| Ensemble (75% Component + 25% Direct) | 29.24 | **62.35** | **0.0652** |

## Key Findings

The component-based approach outperformed direct arrival-delay prediction on the unseen final test set.

Predicting departure deviation and elapsed-time deviation separately produced a lower MAE and higher R² than predicting arrival delay directly.

The ensemble approach provided a small additional improvement in RMSE and R², although the improvement over the component model was limited.

The results suggest that decomposing arrival delay into operational components can provide a more useful modeling structure than treating arrival delay as a single regression target.

Feature importance analysis also showed different patterns across the component models.

For `Elapsed_Time_Deviation`, planned flight duration, geographic information, planned average speed, airport information, and weather variables were among the most influential features.

For `DepDelay` and direct `ArrDelay` prediction, time-of-day, calendar, airport, airline, and weather-related variables were prominent.

## Limitations

Overall R² values remained relatively low.

This indicates that scheduled flight information and pre-departure weather variables alone are not sufficient to explain most of the variation in flight delays.

Important operational information such as aircraft rotation, inbound aircraft delay, crew and gate conditions, airport congestion, and air traffic control conditions was not available in the current dataset.

Therefore, the project results should be interpreted primarily as a comparison of modeling strategies rather than as a production-ready flight delay prediction system.

Feature importance results describe relationships used by the models and should not be interpreted as causal effects.

## Project Structure

```text
flight_delay_prediction/
│
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── .gitignore
├── environment.yml
├── pyproject.toml
└── README.md
```
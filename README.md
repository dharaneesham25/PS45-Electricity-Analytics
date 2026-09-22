# PS45 — Electricity Consumption Analytics & Demand Prediction

Full-stack FastAPI dashboard for electricity consumption analytics, demand forecasting and anomaly detection.

Features:
- Area/category consumption analytics
- Hourly and monthly patterns
- Peak-load detection
- RandomForest demand forecasting with MAE/RMSE/R2
- IsolationForest anomaly detection
- Recommendations
- CSV upload and anomaly export
- Self-contained frontend with no npm build step

Run locally:
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

The demo dataset is generated automatically on startup, so a large CSV does not need to be committed to GitHub.


https://ps45-electricity-analytics-1.onrender.com/

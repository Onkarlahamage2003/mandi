"""
Mandi Price — Live App
=======================
One process that serves the dashboard AND the live data API, so you can run it
on your laptop and open it from your laptop's browser or your phone.

SETUP (one time)
-----------------
1. Install Python 3.9+ if you don't have it.
2. In this folder, run:
       pip install -r requirements.txt
3. Get a free API key from https://data.gov.in/ (Sign up -> My Account -> API Key)
   and set it:
       export DATA_GOV_IN_API_KEY="your_key_here"      (Mac/Linux)
       set DATA_GOV_IN_API_KEY=your_key_here            (Windows cmd)
   Without a key it falls back to the public sample key (slow / rate-limited —
   fine to test, get your own key for real use).

RUN
---
    python app.py

Then:
- On your laptop:  http://localhost:5000
- On your phone (same WiFi as your laptop): find your laptop's local IP
  (Mac: System Settings > WiFi > Details;  Windows: `ipconfig`, look for IPv4)
  then open http://<that-ip>:5000 on your phone's browser.

TO USE IT FROM ANYWHERE (not just home WiFi)
---------------------------------------------
Deploy this same folder to a free host so it gets a permanent public URL:
- Render.com (free web service): connect this folder as a repo, build command
  `pip install -r requirements.txt`, start command `python app.py`.
- Or Railway.app / PythonAnywhere — same idea.
Once deployed, that URL works on any phone or laptop, anywhere, no WiFi sharing needed.
"""

import os
from datetime import timedelta

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

import cache

API_KEY = os.environ.get(
    "DATA_GOV_IN_API_KEY",
    "579b464db66ec23bdd000001cdd3946e44ce4988839e6d2f76b81d6",  # public sample key, rate-limited
)
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

app = Flask(__name__)


def fetch_prices(commodity, state=None, market=None, limit=1000):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
        "filters[commodity]": commodity,
    }
    if state:
        params["filters[state]"] = state
    if market:
        params["filters[market]"] = market

    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    records = resp.json().get("records", [])
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
    for col in ["min_price", "max_price", "modal_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["arrival_date", "modal_price"]).sort_values("arrival_date")
    return df


def forecast_price(daily_series, horizon_days):
    """daily_series: pandas Series indexed by date, modal price. Seasonal+trend blend."""
    if daily_series.empty or len(daily_series) < 10:
        return None

    last_date = daily_series.index[-1]
    last_price = float(daily_series.iloc[-1])

    recent = daily_series.tail(90)
    slope = float(recent.diff().mean()) if len(recent) >= 5 else 0.0
    trend_projection = last_price + slope * horizon_days

    target_date = last_date + timedelta(days=horizon_days)
    window = daily_series[
        (daily_series.index >= target_date - pd.Timedelta(days=372))
        & (daily_series.index <= target_date - pd.Timedelta(days=358))
    ]
    seasonal_projection = float(window.mean()) if not window.empty else last_price

    blended = 0.45 * trend_projection + 0.55 * seasonal_projection
    spread = float(daily_series.tail(180).std() or 0) * 0.6 + max(50.0, last_price * 0.04)

    return {
        "target_date": target_date.strftime("%Y-%m-%d"),
        "forecast_mid": round(blended, 2),
        "forecast_low": round(max(0, blended - spread), 2),
        "forecast_high": round(blended + spread, 2),
        "last_price": round(last_price, 2),
        "as_of": last_date.strftime("%Y-%m-%d"),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/price")
def api_price():
    commodity = request.args.get("commodity", "Onion")
    state = request.args.get("state") or None
    market = request.args.get("market") or None
    horizon = int(request.args.get("horizon", 60))

    try:
        df = fetch_prices(commodity, state, market)
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach data.gov.in: {e}"}), 502

    if df.empty and cache.load_history(commodity).empty:
        return jsonify({"error": f"No records found for '{commodity}'"
                                  f"{' in ' + state if state else ''}"
                                  f"{' at ' + market if market else ''}."}), 404

    # Save today's pull, then merge with everything cached from past runs —
    # the more this app gets used over time, the deeper this history gets.
    if not df.empty:
        cache.save_records(commodity, df)

    cached = cache.load_history(commodity)
    fresh_daily = df.groupby("arrival_date")["modal_price"].mean().rename("modal_price").reset_index() \
        if not df.empty else pd.DataFrame(columns=["arrival_date", "modal_price"])
    fresh_daily = fresh_daily.rename(columns={"arrival_date": "date"})

    merged = pd.concat([cached, fresh_daily]).groupby("date")["modal_price"].mean().sort_index()
    daily = merged.resample("D").mean().interpolate()

    history = [{"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
               for d, p in daily.tail(365).items()]

    fc = forecast_price(daily, horizon)

    return jsonify({
        "commodity": commodity,
        "current_price": history[-1]["price"] if history else None,
        "as_of": history[-1]["date"] if history else None,
        "history": history,
        "forecast": fc,
        "markets_included": sorted(df["market"].dropna().unique().tolist())[:10],
    })


cache.init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render/hosts set PORT themselves
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print("Starting Mandi Price live app...")
    print(f"  Laptop:  http://localhost:{port}")
    print(f"  Phone (same WiFi): http://<your-laptop-local-ip>:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

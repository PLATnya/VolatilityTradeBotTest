#!/usr/bin/env python3

import plotly.graph_objs as go
from trade_tools.okx_trade import OkxClient
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize OKX client
okx_client = OkxClient.from_env()

# Data file to store equity history
EQUITY_DATA_FILE = "data/equity_history.npy"

def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(os.path.dirname(EQUITY_DATA_FILE), exist_ok=True)

def fetch_equity():
    """Fetch current equity from OKX."""
    try:
        balance = okx_client.get_balance()
        total_eq = float(balance.get('totalEq', 0))
        return total_eq
    except Exception as e:
        print(f"Error fetching equity: {e}")
        return None

def save_equity_data(timestamp, equity):
    """Save equity data to file."""
    ensure_data_dir()
    data_point = np.array([[timestamp.timestamp(), equity]], dtype=np.float64)
    
    if not os.path.exists(EQUITY_DATA_FILE):
        np.save(EQUITY_DATA_FILE, data_point)
    else:
        data = np.load(EQUITY_DATA_FILE)
        data = np.vstack([data, data_point])
        np.save(EQUITY_DATA_FILE, data)

def load_equity_data():
    """Load equity data from file."""
    if not os.path.exists(EQUITY_DATA_FILE):
        return [], []
    
    try:
        data = np.load(EQUITY_DATA_FILE)
        timestamps = [datetime.fromtimestamp(ts) for ts in data[:, 0]]
        equity_values = data[:, 1].tolist()
        return timestamps, equity_values
    except Exception as e:
        print(f"Error loading equity data: {e}")
        return [], []


def update_equity_internal():
    equity = fetch_equity()

    if equity is not None:
        current_time = datetime.now()
        save_equity_data(current_time, equity)
        last_update_text = f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Current Equity: {equity:.2f} USDT"
    else:
        last_update_text = f"Last update attempt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Error fetching equity"
    return last_update_text


# Fetch initial equity on startup
if __name__ == '__main__':
    # Fetch initial equity

    # INSERT_YOUR_CODE
    import argparse

    parser = argparse.ArgumentParser(description='Volatility dashboard equity tracker')
    parser.add_argument('--delete-old', action='store_true', help='Delete old equity data file before running')
    args = parser.parse_args()

    if args.delete_old and os.path.exists(EQUITY_DATA_FILE):
        os.remove(EQUITY_DATA_FILE)
        print("Deleted old equity data file.")


    update_equity_internal()
    timestamps, equity_values = load_equity_data()

    fig, ax = plt.subplots()
    while True:
        line1, = ax.plot(timestamps, equity_values, linestyle='-', label='Sample1')
        plt.pause(1800)
        line1.remove()
        update_equity_internal()
        timestamps, equity_values = load_equity_data()
        print(len(equity_values))
    # timestamps, equity_values = [], []
    # update_equity_internal()
    # plt.ion()

    # figure, ax = plt.subplots(figsize=(8, 6))
    # (line1,) = ax.plot(timestamps, equity_values)

    # plt.title("Dynamic Plot of equity", fontsize=25)

    # plt.xlabel("Time", fontsize=18)
    # plt.ylabel("Equity", fontsize=18)

    # while True:
    #     timestamps, equity_values = load_equity_data()
    #     update_equity_internal()
        
    #     line1.set_xdata(timestamps)
    #     line1.set_ydata(equity_values)

    #     figure.canvas.draw()

    #     figure.canvas.flush_events()
    #     time.sleep(1)
    
    





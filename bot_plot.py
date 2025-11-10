
import matplotlib.pyplot as plt
import plotext as pltcli
import numpy as np
import os
import json

CONFIG_PATH = ""
CONFIG = None
REALTIME_DATA_FILE = ""
BEST_TRIALS_FILE = ""
def reload_config():
    global CONFIG_PATH
    load_config(CONFIG_PATH)

def load_config(config_path: str):
    global CONFIG_PATH
    CONFIG_PATH = config_path
    global CONFIG
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

    global REALTIME_DATA_FILE
    REALTIME_DATA_FILE = f"data/{CONFIG['instrument_id']}_{CONFIG['timeframe']}_realtime_data.npy"

    global BEST_TRIALS_FILE
    BEST_TRIALS_FILE = f"{CONFIG['instrument_id']}_{CONFIG['timeframe']}_best_trials.json"

class BacktestInfo:
    def __init__(self):
        self.equity_array = []
        self.price_array = []
        self.upper_trade_treshold = 0
        self.lower_trade_treshold = 0
        self.long_term_window = 0
        self.short_term_window = 0
        self.volatility_ratio_array = []

    @staticmethod
    def from_dict(data):
        obj = BacktestInfo()
        obj.equity_array = data.get("equity_array", [])
        obj.price_array = data.get("price_array", [])
        obj.upper_trade_treshold = data.get("upper_trade_treshold", 0)
        obj.lower_trade_treshold = data.get("lower_trade_treshold", 0)
        obj.long_term_window = data.get("long_term_window", 0)
        obj.short_term_window = data.get("short_term_window", 0)
        obj.volatility_ratio_array = data.get("volatility_ratio_array", [])
        return obj


def plot_backtest_info_cli(backtest_infos):
    """
    Plot arrays of BacktestInfo instances in shared subplots.
    Each BacktestInfo is displayed in its own column (subplot column).
    """
    n = len(backtest_infos)

    pltcli.clf()
    pltcli.subplots(2, n)
    #pltcli.title(f"Long term window: {backtest_infos[0].long_term_window}, Short term window: {backtest_infos[0].short_term_window}")
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    colors = colors[:n]
    for idx, backtest_info in enumerate(backtest_infos):
        # Equity Curve - each in its own column
        ax_eq = pltcli.subplot(1, idx + 1)
        ax_eq.plot(backtest_info.equity_array, marker = "braille", color="red")
        ax_eq.title(f"Equity long {backtest_info.long_term_window} short {backtest_info.short_term_window} upper {backtest_info.upper_trade_treshold} lower {backtest_info.lower_trade_treshold}")

        # Volatility Ratio - each in its own column
        ax_vr = pltcli.subplot(2, idx + 1)
        ax_vr.plot(backtest_info.volatility_ratio_array, marker = "braille", color="blue", label='Volatility Ratio')

        # Map close prices to [0, 1] for the same plot
        close_prices_plot = backtest_info.price_array
        cp_min = np.min(close_prices_plot)
        cp_max = np.max(close_prices_plot)
        if cp_max != cp_min:
            close_prices_norm = (close_prices_plot - cp_min) / (cp_max - cp_min)
        else:
            close_prices_norm = np.zeros_like(close_prices_plot)

        vrr = np.array(backtest_info.volatility_ratio_array)
        vrr_min = vrr.min()
        vrr_max = vrr.max()
        if vrr_max != vrr_min:
            mapped_close = close_prices_norm * (vrr_max - vrr_min) + vrr_min
        else:
            mapped_close = close_prices_norm + vrr_min

        ax_vr.plot(mapped_close, marker = "braille", color='red', label="Close Price (mapped)")

        # Add reference lines
        print(f'upper_trade_treshold ({backtest_info.upper_trade_treshold})')
        print(f'lower_trade_treshold ({backtest_info.lower_trade_treshold})')
        ax_vr.hline(backtest_info.upper_trade_treshold, color='black')
        ax_vr.hline(backtest_info.lower_trade_treshold, color='black')

    pltcli.show()

def plot_backtest_info(backtest_info, cli = False):
    if cli:
        plot_backtest_info_cli(backtest_info)
    else:
        plot_backtest_info_ui(backtest_info)

def plot_backtest_info_ui(backtest_infos):
    """
    Plot arrays of BacktestInfo instances in shared subplots.
    Each BacktestInfo is displayed in its own column (subplot column).
    """
    n = len(backtest_infos)

    fig, axs = plt.subplots(2, n, figsize=(6 * n, 8), squeeze=False)

    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for idx, (backtest_info, color) in enumerate(zip(backtest_infos, colors)):
        # Equity Curve - each in its own column
        ax_eq = axs[0, idx]
        ax_eq.plot(backtest_info.equity_array, color=color)
        ax_eq.set_title(f"Equity Curve {idx+1} (long_term_window: {backtest_info.long_term_window}, short_term_window: {backtest_info.short_term_window})")

        # Volatility Ratio - each in its own column
        ax_vr = axs[1, idx]
        ax_vr.plot(backtest_info.volatility_ratio_array, color=color, label='Volatility Ratio')

        # Map close prices to [0, 1] for the same plot
        close_prices_plot = backtest_info.price_array
        cp_min = np.min(close_prices_plot)
        cp_max = np.max(close_prices_plot)
        if cp_max != cp_min:
            close_prices_norm = (close_prices_plot - cp_min) / (cp_max - cp_min)
        else:
            close_prices_norm = np.zeros_like(close_prices_plot)

        vrr = np.array(backtest_info.volatility_ratio_array)
        vrr_min = vrr.min()
        vrr_max = vrr.max()
        if vrr_max != vrr_min:
            mapped_close = close_prices_norm * (vrr_max - vrr_min) + vrr_min
        else:
            mapped_close = close_prices_norm + vrr_min

        ax_vr.plot(mapped_close, color=color, linestyle="dashed", alpha=0.5, label="Close Price (mapped)")

        # Add reference lines
        ax_vr.axhline(y=backtest_info.upper_trade_treshold, color='r', linestyle='--', label=f'upper_trade_treshold ({backtest_info.upper_trade_treshold})')
        ax_vr.axhline(y=backtest_info.lower_trade_treshold, color='g', linestyle='--', label=f'lower_trade_treshold ({backtest_info.lower_trade_treshold})')

        # Legends
        ax_eq.legend([f'Equity'])
        ax_vr.legend()

    plt.tight_layout()
    plt.show()


def plot_equity_data(cli = False):
    if cli:
        plot_equity_data_cli()
    else:
        plot_equity_data_ui()


def plot_equity_data_ui():
    global REALTIME_DATA_FILE
    data = np.load(REALTIME_DATA_FILE)
    plt.plot(data[:, 3], label='Equity')
    plt.show()

def plot_equity_data_cli():
    global REALTIME_DATA_FILE
    data = np.load(REALTIME_DATA_FILE)
    pltcli.plot(data[:, 3], label='Equity')
    pltcli.show()


def plot_realtime_data_ui():
    global REALTIME_DATA_FILE
    if not os.path.exists(REALTIME_DATA_FILE):
        print(f"File {REALTIME_DATA_FILE} does not exist.")
        return

    data = np.load(REALTIME_DATA_FILE)
    ratio = data[:,0]
    price = data[:,1]

    # Normalize price to ratio min-max range
    ratio_min = np.min(ratio)
    ratio_max = np.max(ratio)
    price_min = np.min(price)
    price_max = np.max(price)
    if price_max != price_min:
        price_norm = (price - price_min) / (price_max - price_min)
        price_scaled = price_norm * (ratio_max - ratio_min) + ratio_min
    else:
        price_scaled = np.full_like(price, ratio_min)

    plt.plot(ratio, label='Volatility Ratio')
    plt.plot(price_scaled, label='Price')


    plt.axhline(y=CONFIG["real_time_trade"]["upper_trade_treshold"], color='orange', linestyle='--', label='Upper Threshold')
    plt.axhline(y=CONFIG["real_time_trade"]["lower_trade_treshold"], color='purple', linestyle='--', label='Lower Threshold')
    plt.legend()
    plt.show()


def plot_realtime_data_cli():
    global REALTIME_DATA_FILE
    if not os.path.exists(REALTIME_DATA_FILE):
        print(f"File {REALTIME_DATA_FILE} does not exist.")
        return

    data = np.load(REALTIME_DATA_FILE)
    ratio = data[:, 0]
    price = data[:, 1]

    ratio_min = np.min(ratio)
    ratio_max = np.max(ratio)
    price_min = np.min(price)
    price_max = np.max(price)
    if price_max != price_min:
        price_norm = (price - price_min) / (price_max - price_min)
        price_scaled = price_norm * (ratio_max - ratio_min) + ratio_min
    else:
        price_scaled = np.full_like(price, ratio_min)

    pltcli.clf()
    pltcli.plot(ratio, label='Volatility Ratio', color='blue')
    pltcli.plot(price_scaled, label='Price', color='orange')

    # Add threshold horizontal lines
    upper_th = CONFIG["real_time_trade"]["upper_trade_treshold"]
    lower_th = CONFIG["real_time_trade"]["lower_trade_treshold"]
    pltcli.horizontal_line(upper_th, color='orange')
    pltcli.horizontal_line(lower_th, color='magenta')

    pltcli.title("Real-time Volatility Ratio and Price")
    pltcli.show()


def plot_realtime_data(cli = False):
    if cli:
        plot_realtime_data_cli()
    else:
        plot_realtime_data_ui()


def plot_best_trials():
    with open(BEST_TRIALS_FILE, "r") as f:
        best_trials_data = json.load(f)

    for trial in best_trials_data:
        print(trial['params'])
        print("--------------------------------")
        backtest_infos = [BacktestInfo.from_dict(i) for i in trial['backtest_infos']]
        plot_backtest_info(backtest_infos, cli=CONFIG["cli_plot"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bot plot script")
    parser.add_argument("-c", "--config", type=str, default="volatility_config.json", help="Path to JSON config file")
    parser.add_argument("-r", "--realtime", action="store_true", help="Realtime data")
    parser.add_argument("-b", "--best-trials", action="store_true", help="Best trials")
    parser.add_argument("-e", "--equity", action="store_true", help="Realtime data")
    args = parser.parse_args()

    load_config(args.config)

    if args.best_trials:
        plot_best_trials()
        
    if args.realtime:
        plot_realtime_data(CONFIG["cli_plot"])

    if args.equity:
        plot_equity_data(CONFIG["cli_plot"])
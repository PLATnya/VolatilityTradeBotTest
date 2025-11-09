import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from okx_trade import OkxClient
from dotenv import load_dotenv
import os
import time
import optuna
import json
from datetime import datetime, timedelta, timezone

load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")

OKX = OkxClient.from_env()


CONFIG = None
def convert_time_to_ms(time: str) -> str:
    # Convert "ddmmyyyy_ssmmhh" to ms timestamp
    # Example: '01052023_300412' -> 1 May 2023, 12:04:30
    time_ts = 0
    try:
        date_part, time_part = time.split('_')
        day = int(date_part[:2])
        month = int(date_part[2:4])
        year = int(date_part[4:8])
        hour = int(time_part[:2])
        min_ = int(time_part[2:4])
        sec = int(time_part[4:6])
        from datetime import datetime, timezone
        dt = datetime(year, month, day, hour, min_, sec, tzinfo=timezone.utc)
        time_ts = str(int(dt.timestamp() * 1000))
    except Exception as e:
        raise ValueError(f"Invalid format: ({e})")

    return time_ts

def load_candles(interval="1h", limit=1000, symbol="SOL-USDT", start_time=None):
    """
    Load candles from the Binance public API.
    Returns DataFrame: timestamp, open, high, low, close, volume, etc.
    """
    # Binance expects symbols like SOLUSDT, intervals as 1m, 1h, 4h, 1d, etc.
    #binance_symbol = symbol
    binance_symbol = symbol.replace("-", "")

    if start_time is not None:
        start_time_ms = convert_time_to_ms(start_time)
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={binance_symbol}&interval={interval}&limit={limit}&startTime={start_time_ms}"
        )
    else:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={binance_symbol}&interval={interval}&limit={limit}"
        )
    resp = requests.get(url)
    #resp.raise_for_status()
    data = resp.json()

    # Data is a list of lists, each representing a kline:
    # [ open time, open, high, low, close, volume, close time, quote asset volume, number of trades, taker base volume, taker quote volume, ignore ]
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_base_volume', 'taker_quote_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].astype(float)
    return df[cols]

def load_candles_all(interval="1h", symbol="SOL-USDT", start_time=None):
    candles = load_candles(interval=interval, limit=1000, symbol=symbol, start_time=start_time)

    def get_last_index_str(data):
        last_index = data.index[-1]
        last_index_str = ""
        if hasattr(last_index, 'strftime'):
            # Convert to string in format "ddmmyyyy_ssmmhh"
            last_index_str = last_index.strftime("%d%m%Y_%H%M%S")
        else:
            last_index_str = str(last_index)

        return last_index_str

    while True:
        candles2 = load_candles(interval=interval, limit=1000, symbol=symbol, start_time=get_last_index_str(candles)).iloc[1:]
        if candles2.empty:
            break
        candles = pd.concat([candles, candles2])

    return candles


def calculate_volatility_ratio(in_data, long_term_window: int, short_term_window: int):
    close_prices = in_data['close']
    returns = close_prices.pct_change().dropna()

    # Volatility ratio: Current volatility divided by volatility of previous 20 periods (long-term/short-term ratio)
    long_term_vol = returns.rolling(window=long_term_window).std()  # You can adjust 50 for a longer-term representation
    short_term_vol = returns.rolling(window=short_term_window).std()
    return short_term_vol / long_term_vol


class BacktestInfo:
    def __init__(self):
        self.equity_array = []
        self.price_array = []
        self.upper_trade_treshold = 0
        self.lower_trade_treshold = 0
        self.volatility_ratio_array = []

    @staticmethod
    def from_dict(data):
        obj = BacktestInfo()
        obj.equity_array = data.get("equity_array", [])
        obj.price_array = data.get("price_array", [])
        obj.upper_trade_treshold = data.get("upper_trade_treshold", 0)
        obj.lower_trade_treshold = data.get("lower_trade_treshold", 0)
        obj.volatility_ratio_array = data.get("volatility_ratio_array", [])
        return obj

def hollow_backtest_period(in_data, long_term_window: int, short_term_window: int, upper_trade_treshold: float, lower_trade_treshold: float, plot: bool = False):
    open_short = False
    open_long = False

    last_price = 0

    equity = 0
    some_array = [0]
    volatility_ratio_array = []
    for i in range(long_term_window + 1, len(in_data)):
        df = in_data.iloc[i-(long_term_window + 1):i]

        volatility_ratio = calculate_volatility_ratio(df, long_term_window, short_term_window)
        volatility_ratio_array.append(volatility_ratio.iloc[-1])

        price = df['close'].iloc[-1]

        profit = 0
        if volatility_ratio.iloc[-1] > upper_trade_treshold and not open_short:
            open_long = False
            open_short = True
            #print("open short")
            if last_price != 0:
                profit = price/last_price - 1 - 0.001
                #print(f"profit: {profit}")
            last_price = price
            #print(f"Volatility ratio: {volatility_ratio} at {df.index[-1]}")
        if volatility_ratio.iloc[-1] < lower_trade_treshold and not open_long:
            open_long = True
            open_short = False
            #print("open long")
            if last_price != 0:
                profit = last_price/price - 1- 0.001
                #print(f"profit: {profit}")
            last_price = price

        equity += profit
        some_array.append(equity)


    backtest_info = BacktestInfo()
    backtest_info.equity_array = some_array
    backtest_info.price_array = in_data['close'].iloc[(long_term_window + 1):].values
    backtest_info.upper_trade_treshold = upper_trade_treshold
    backtest_info.lower_trade_treshold = lower_trade_treshold
    backtest_info.volatility_ratio_array = volatility_ratio_array

    if plot:
        plot_backtest_info(backtest_info)
        
    changes = pd.Series(some_array).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    changes_mean = float('inf')
    if len(changes) > 0:
        changes_mean = changes.unique().mean()

    return equity, changes_mean, backtest_info


def plot_backtest_info(backtest_infos):
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
        ax_eq.set_title(f"Equity Curve {idx+1}")

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

    
def calculate_current_volatility_ratio():
    global CONFIG
    candles = load_candles(interval=CONFIG['timeframe'], limit=CONFIG['real_time_trade']['long_term_window'] + 1, symbol=CONFIG['instrument_id'])
    return calculate_volatility_ratio(candles, CONFIG['real_time_trade']['long_term_window'], CONFIG['real_time_trade']['short_term_window']).iloc[-1], candles['close'].iloc[-1]



def get_days_ago_str(days: int):

    now = datetime.now(timezone.utc)
    one_month_ago = now - timedelta(days=days)
    return one_month_ago.strftime("%d%m%Y_%H%M%S")


def load_config(config_path: str):
    global CONFIG
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

def optimize_parameters():
    global CONFIG
    backtest_infos = []
    n_trials = CONFIG["optimization"]["n_trials"]
    days_list = CONFIG["optimization"]["days_list"]
    def objective(trial):
        global CONFIG
        """Optimize only long_term_window_trade and short_term_window_trade."""
        long_term_window_range = CONFIG["long_term_window_range"]
        short_term_window_range = CONFIG["short_term_window_range"]
        upper_trade_treshold_range = CONFIG["upper_trade_treshold_range"]
        lower_trade_treshold_range = CONFIG["lower_trade_treshold_range"]

        upper_trade_treshold = trial.suggest_float('upper_trade_treshold', upper_trade_treshold_range[0], upper_trade_treshold_range[1], step=upper_trade_treshold_range[2])
        lower_trade_treshold = trial.suggest_float('lower_trade_treshold', lower_trade_treshold_range[0], lower_trade_treshold_range[1], step=lower_trade_treshold_range[2])
        long_term_window = trial.suggest_int('long_term_window' , long_term_window_range[0], long_term_window_range[1], step=long_term_window_range[2])
        short_term_window = trial.suggest_int('short_term_window', short_term_window_range[0], short_term_window_range[1], step=short_term_window_range[2])

        # Use fixed values for other parameters
        backtest_infos.append([])
        equity_list = []
        changes_mean_list = []
        for days in days_list:
            in_data = load_candles_all(interval=CONFIG['timeframe'], start_time=get_days_ago_str(days), symbol=CONFIG['instrument_id'])
            equity, changes_mean, backtest_info = hollow_backtest_period(
                in_data, 
                long_term_window=long_term_window,  # Fixed
                short_term_window=short_term_window,  # Fixed
                upper_trade_treshold=upper_trade_treshold,
                lower_trade_treshold=lower_trade_treshold,
                plot=False
            )
            backtest_infos[-1].append(backtest_info)
            equity_list.append(equity)
            changes_mean_list.append(changes_mean)
        return [item for item in zip(equity_list, changes_mean_list) for item in item]

    study = optuna.create_study(directions=['maximize', 'minimize'] * len(days_list))
    study.optimize(objective, n_trials=n_trials)
    
    best_trials_data = []
    for trial in study.best_trials:
        trial_dict = {
            "params": trial.params,
            "values": trial.values,
            "backtest_infos": [{
                "equity_array": list(backtest_info.equity_array),
                "price_array": list(backtest_info.price_array),
                "upper_trade_treshold": backtest_info.upper_trade_treshold,
                "lower_trade_treshold": backtest_info.lower_trade_treshold,
                "volatility_ratio_array": list(backtest_info.volatility_ratio_array),
            } for backtest_info in backtest_infos[trial.number]]
        }

        best_trials_data.append(trial_dict)

    with open(f"{CONFIG['instrument_id']}_{CONFIG['timeframe']}_best_trials.json", "w") as f:
        json.dump(best_trials_data, f, indent=4)

    print("")
    print("--------------------------------")
    for i in study.best_trials:
        print(i.params)
        print(i.values)
        print("--------------------------------")

def plot_best_trials(best_trials_file: str):
    with open(best_trials_file, "r") as f:
        best_trials_data = json.load(f)

    for trial in best_trials_data:
        print(trial['params'])
        print("--------------------------------")
        backtest_infos = [BacktestInfo.from_dict(i) for i in trial['backtest_infos']]
        plot_backtest_info(backtest_infos)

def simple_test_launch():
    global CONFIG
    print(CONFIG)
    #long_term_window = int(CONFIG["long_term_window_range"][0] +(CONFIG["long_term_window_range"][1] - CONFIG["long_term_window_range"][0])/2) 
    #short_term_window = int(CONFIG["short_term_window_range"][0] +(CONFIG["short_term_window_range"][1] - CONFIG["short_term_window_range"][0])/2)
    #upper_trade_treshold = float(CONFIG["upper_trade_treshold_range"][0] +(CONFIG["upper_trade_treshold_range"][1] - CONFIG["upper_trade_treshold_range"][0])/2)
    #lower_trade_treshold = float(CONFIG["lower_trade_treshold_range"][0] +(CONFIG["lower_trade_treshold_range"][1] - CONFIG["lower_trade_treshold_range"][0])/2)
    long_term_window = CONFIG["real_time_trade"]["long_term_window"]
    short_term_window = CONFIG["real_time_trade"]["short_term_window"]
    lower_trade_treshold = CONFIG["real_time_trade"]["lower_trade_treshold"]
    upper_trade_treshold = CONFIG["real_time_trade"]["upper_trade_treshold"]
    in_data = load_candles_all(interval=CONFIG["timeframe"], symbol=CONFIG["instrument_id"], start_time=get_days_ago_str(CONFIG["optimization"]["days_list"][0]))
    equity, changes_mean, backtest_info = hollow_backtest_period(
        in_data, 
        long_term_window=long_term_window ,  
        short_term_window=short_term_window,  
        upper_trade_treshold=upper_trade_treshold,
        lower_trade_treshold=lower_trade_treshold,
        plot=False
    )
    in_data = load_candles_all(interval=CONFIG["timeframe"], symbol=CONFIG["instrument_id"], start_time=get_days_ago_str(CONFIG["optimization"]["days_list"][1]))

    equity, changes_mean, backtest_info2 = hollow_backtest_period(
        in_data, 
        long_term_window=long_term_window ,  
        short_term_window=short_term_window,  
        upper_trade_treshold=upper_trade_treshold,
        lower_trade_treshold=lower_trade_treshold,
        plot=False
    )
    #print(equity, changes_mean)
    plot_backtest_info([backtest_info, backtest_info2])


def get_order_size():
    global CONFIG
    return CONFIG["real_time_trade"]["order_size"]/float(OKX.get_ticker(f"{CONFIG["instrument_id"]}-SWAP")['last'])

def write_realtime_data(volatility_ratio: float, price:float, order_direction: int = 0):
    print(f"writing realtime data: {volatility_ratio}, {price}, {order_direction}")
    global CONFIG
    file_name = f"data/volatility/{CONFIG['instrument_id']}_{CONFIG['timeframe']}_realtime_data.npy"
    if not os.path.exists(file_name):
        np.save(file_name, np.array([[volatility_ratio, price, order_direction]], dtype=np.float64))
    else:
        data = np.load(file_name)
        data = np.vstack([data, np.array([[volatility_ratio, price, order_direction]])])
        np.save(file_name, data)

def real_time_trade():
    """
    Executes `func` every `n` seconds minus function execution time.
    Passes *args and **kwargs to func.
    """
    global CONFIG
    n = CONFIG["real_time_trade"]["delay_sec"]

    is_long_opened = False
    is_short_opened = False
    while True:
        start_time = time.time()
        volatility_ratio, price = calculate_current_volatility_ratio()
        instrument_id = f"{CONFIG["instrument_id"]}-SWAP"
        write_realtime_data(volatility_ratio, price, 0)
        if volatility_ratio > CONFIG["real_time_trade"]["upper_trade_treshold"] and not is_short_opened:
            print("open long")

            if is_long_opened:
                OKX.close_position(inst_id=instrument_id, mgn_mode="cross", pos_side="long", auto_cxl=True)
                is_long_opened = False

            order_size = get_order_size()
            OKX.place_order(instrument_id, "cross", "sell", "market", f"{order_size:.2f}", position_side="short")
            is_short_opened = True

        elif volatility_ratio < CONFIG["real_time_trade"]["lower_trade_treshold"] and not is_long_opened:
            print("open short")

            if is_short_opened:
                OKX.close_position(inst_id=instrument_id, mgn_mode="cross", pos_side="short", auto_cxl=True)
                is_short_opened = False
                
            order_size = get_order_size()
            OKX.place_order(instrument_id, "cross", "buy", "market", f"{order_size:.2f}", position_side="long")
            is_long_opened = True

        elapsed = time.time() - start_time
        to_sleep = max(0, n - elapsed)
        time.sleep(to_sleep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Volatility trade script")
    parser.add_argument("-c", "--config", type=str, default="volatility_config.json", help="Path to JSON config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    config_path = args.config
    load_config(config_path)

    print(CONFIG["instrument_id"])
    print(CONFIG["timeframe"])

    #simple_test_launch()
    #plot_best_trials(f"{CONFIG['instrument_id']}_{CONFIG['timeframe']}_best_trials.json")
    if args.debug:
        simple_test_launch()
    else:
        real_time_trade()


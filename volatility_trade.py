import pandas as pd
import requests
import numpy as np
from okx_trade import OkxClient
import plotext as plt
from dotenv import load_dotenv
import os
import time
import optuna
import json
from datetime import datetime, timedelta, timezone
from bot_plot import plot_backtest_info, BacktestInfo

load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")

OKX = OkxClient.from_env()

CONFIG_PATH = ""
CONFIG = None

REALTIME_DATA_FILE = ""
BEST_TRIALS_FILE = ""

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


def hollow_backtest_period(in_data, long_term_window: int, short_term_window: int, upper_trade_treshold: float, lower_trade_treshold: float, plot: bool = False):
    open_short = False
    open_long = False

    last_price = 0

    equity = 0
    some_array = [0]
    volatility_ratio_array = []

    global CONFIG
    treshold_window = CONFIG["real_time_trade"]["treshold_window"]
    print(f"treshold window: {treshold_window}")

    treshold_window_upper_array = []
    treshold_window_lower_array = []

    for i in range(max(long_term_window + 1, treshold_window + 1), len(in_data)):
        df = in_data.iloc[i-(long_term_window + 1):i]

        volatility_ratio = calculate_volatility_ratio(df, long_term_window, short_term_window)
        volatility_ratio_array.append(volatility_ratio.iloc[-1])

        price = df['close'].iloc[-1]

        profit = 0

        if len(volatility_ratio_array) > treshold_window + 2:

            #volatility_ratio.iloc[-1] > upper_trade_treshold and not open_short
            #volatility_ratio.iloc[-1] < lower_trade_treshold and not open_long

            max_ratio = volatility_ratio_array[-2]
            min_ratio = volatility_ratio_array[-2]
            for i in range(treshold_window):
                if volatility_ratio_array[-(2+i)] > max_ratio:
                    max_ratio = volatility_ratio_array[-(2+i)]
                if volatility_ratio_array[-(2+i)] < min_ratio:
                    min_ratio = volatility_ratio_array[-(2+i)]

            print(f"max ratio: {max_ratio}, min ratio: {min_ratio} ratio: {volatility_ratio.iloc[-1]}")
            upper_condition = volatility_ratio.iloc[-1] > max_ratio and (not open_short)
            lower_condition = volatility_ratio.iloc[-1] < min_ratio and (not open_long)

            #treshold_window_upper_array.append(volatility_ratio.rolling(treshold_window).max().iloc[-1])
            #treshold_window_lower_array.append(volatility_ratio.rolling(treshold_window).min().iloc[-1])

            if upper_condition:
                open_long = False
                open_short = True
                print(f"open short at {df.index[-1]}")
                if last_price != 0:
                    profit = price/last_price - 1 - 0.001
                last_price = price

            if lower_condition:
                open_long = True
                open_short = False
                if last_price != 0:
                    profit = last_price/price - 1- 0.001
                last_price = price

        equity += profit
        some_array.append(equity)

    backtest_info = BacktestInfo()
    backtest_info.long_term_window = long_term_window
    backtest_info.short_term_window = short_term_window
    backtest_info.equity_array = some_array
    backtest_info.price_array = in_data['close'].iloc[(long_term_window + 1):].values
    backtest_info.upper_trade_treshold = upper_trade_treshold
    backtest_info.lower_trade_treshold = lower_trade_treshold
    backtest_info.volatility_ratio_array = volatility_ratio_array

    if plot:
        plot_backtest_info(backtest_info, cli=CONFIG["cli_plot"])
        
    changes = pd.Series(some_array).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    changes_mean = float('inf')
    if len(changes) > 0:
        changes_mean = changes.unique().mean()

    return equity, changes_mean, backtest_info
    
def calculate_current_volatility_ratio():
    global CONFIG
    candles = load_candles(interval=CONFIG['timeframe'], limit=CONFIG['real_time_trade']['long_term_window'] + 1, symbol=CONFIG['instrument_id'])
    return calculate_volatility_ratio(candles, CONFIG['real_time_trade']['long_term_window'], CONFIG['real_time_trade']['short_term_window']).iloc[-1], candles['close'].iloc[-1]


def get_days_ago_str(days: int):

    now = datetime.now(timezone.utc)
    one_month_ago = now - timedelta(days=days)
    return one_month_ago.strftime("%d%m%Y_%H%M%S")

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

def optimize_parameters():
    reload_config()
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
                "long_term_window": backtest_info.long_term_window,
                "short_term_window": backtest_info.short_term_window,
            } for backtest_info in backtest_infos[trial.number]]
        }

        best_trials_data.append(trial_dict)

    with open(BEST_TRIALS_FILE, "w") as f:
        json.dump(best_trials_data, f, indent=4)

    print("")
    print("--------------------------------")
    for i in study.best_trials:
        print(i.params)
        print(i.values)
        print("--------------------------------")

def optimize_parameters_only_equity():
    reload_config()
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
        return equity_list

    study = optuna.create_study(directions=['maximize',] * len(days_list))
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
                "long_term_window": backtest_info.long_term_window,
                "short_term_window": backtest_info.short_term_window,
            } for backtest_info in backtest_infos[trial.number]]
        }

        best_trials_data.append(trial_dict)

    with open(BEST_TRIALS_FILE, "w") as f:
        json.dump(best_trials_data, f, indent=4)

    print("")
    print("--------------------------------")
    for i in study.best_trials:
        print(i.params)
        print(i.values)
        print("--------------------------------")

def simple_test_launch():
    global CONFIG
    reload_config()
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

    plot_backtest_info([backtest_info, backtest_info2], cli=CONFIG["cli_plot"])


def get_order_size():
    global CONFIG
    return CONFIG["real_time_trade"]["order_size"]/float(OKX.get_ticker(f"{CONFIG["instrument_id"]}-SWAP")['last'])

def write_realtime_data(volatility_ratio: float, price:float, order_direction: int = 0, equity: float = None):
    print(f"writing realtime data: {volatility_ratio}, {price}, {order_direction}")
    global CONFIG
    file_name = f"data/{CONFIG['instrument_id']}_{CONFIG['timeframe']}_realtime_data.npy"
    if not os.path.exists(file_name):
        if not equity:
            equity = 0 
        np.save(file_name, np.array([[volatility_ratio, price, order_direction, equity]], dtype=np.float64))
    else:
        data = np.load(file_name)
        if not equity:
            equity = data[-1, 3]
        data = np.vstack([data, np.array([[volatility_ratio, price, order_direction, equity]])])
        np.save(file_name, data)

def fetch_equity():
    """Fetch current equity from OKX."""
    try:
        balance = OKX.get_balance()
        total_eq = float(balance.get('totalEq', 0))
        return total_eq
    except Exception as e:
        print(f"Error fetching equity: {e}")
        return None

IS_LONG_OPENED = False
IS_SHORT_OPENED = False

TRADE_ATTEMPTS = 5
def open_long(instrument_id: str):
    order_size = get_order_size()
    print("open long, order size: ", order_size)
    for i in range(TRADE_ATTEMPTS):
        try:
            OKX.place_order(instrument_id, "cross", "buy", "market", f"{order_size:.2f}", position_side="long")
        except Exception as e:
            print(f"Error opening long: {e}")
            time.sleep(1)
            continue
        break
def open_short(instrument_id: str):
    order_size = get_order_size()
    print("open short, order size: ", order_size)
    for i in range(TRADE_ATTEMPTS):
        try:
            OKX.place_order(instrument_id, "cross", "sell", "market", f"{order_size:.2f}", position_side="short")
        except Exception as e:
            print(f"Error opening short: {e}")
            time.sleep(1)
            continue
        break

def close_long(instrument_id: str):
    print("close long")
    for i in range(TRADE_ATTEMPTS):
        try:
            OKX.close_position(inst_id=instrument_id, mgn_mode="cross", pos_side="long", auto_cxl=True)
        except Exception as e:
            print(f"Error closing long: {e}")
            time.sleep(1)
            continue
        break

def close_short(instrument_id: str):
    print("close short")
    for i in range(TRADE_ATTEMPTS):
        try:
            OKX.close_position(inst_id=instrument_id, mgn_mode="cross", pos_side="short", auto_cxl=True)
        except Exception as e:
            print(f"Error closing short: {e}")
            time.sleep(1)
            continue
        break

def real_time_trade_step():
    global IS_LONG_OPENED
    global IS_SHORT_OPENED
    reload_config()
    volatility_ratio, price = calculate_current_volatility_ratio()
    instrument_id = f"{CONFIG["instrument_id"]}-SWAP"
    
    order_direction = 0
    if volatility_ratio > CONFIG["real_time_trade"]["upper_trade_treshold"] and not IS_SHORT_OPENED:
        if IS_LONG_OPENED:
            close_long(instrument_id)
            IS_LONG_OPENED = False

        open_short(instrument_id)
        IS_SHORT_OPENED = True
        order_direction = -1

    elif volatility_ratio < CONFIG["real_time_trade"]["lower_trade_treshold"] and not IS_LONG_OPENED:
        if IS_SHORT_OPENED:
            close_short(instrument_id)
            IS_SHORT_OPENED = False
            
        open_long(instrument_id)
        IS_LONG_OPENED = True
        order_direction = 1

    equity = fetch_equity()
    write_realtime_data(volatility_ratio, price, order_direction, equity)

def real_time_trade():
    """
    Executes `func` every `n` seconds minus function execution time.
    Passes *args and **kwargs to func.
    """
    global CONFIG
    n = CONFIG["real_time_trade"]["delay_sec"]

    while True:
        start_time = time.time()
        real_time_trade_step()

        elapsed = time.time() - start_time
        to_sleep = max(0, n - elapsed)
        
        sleep_time = to_sleep
        while True:
            mins, secs = divmod(int(sleep_time), 60)
            print(f"Sleeping for {mins} minutes and {secs} seconds...")
            time.sleep(1)
            sleep_time -= 1
            if sleep_time <= 0:
                break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Volatility trade script")
    parser.add_argument("-c", "--config", type=str, default="volatility_config.json", help="Path to JSON config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-o", "--optimize", action="store_true", help="Optimize parameters")
    parser.add_argument("-oe", "--optimize-only-equity", action="store_true", help="Optimize parameters only equity")

    args = parser.parse_args()

    load_config(args.config)

    print(CONFIG["instrument_id"])
    print(CONFIG["timeframe"])

    if not os.path.exists("data"):
        os.makedirs("data")

    if args.optimize:
        optimize_parameters()
        exit()

    if args.optimize_only_equity:
        optimize_parameters_only_equity()
        exit()

    if args.debug:
        simple_test_launch()
    else:
        real_time_trade()


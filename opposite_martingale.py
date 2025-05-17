import time, math, asyncio, json, threading
from datetime import datetime
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib

global_value.loglevel = 'INFO'

# Session configuration
start_counter = time.perf_counter()

### REAL SSID Format::
#ssid = """42["auth",{"session":"a:4:{s:10:\\"session_id\\";s:32:\\"aa11b2345c67d89e0f1g23456h78i9jk\\";s:10:\\"ip_address\\";s:11:\\"11.11.11.11\\";s:10:\\"user_agent\\";s:111:\\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36\\";s:13:\\"last_activity\\";i:1234567890;}1234a5b678901cd2efghi34j5678kl90","isDemo":0,"uid":12345678,"platform":2}]"""
#demo = False

### DEMO SSID Format::
#ssid = """42["auth",{"session":"abcdefghijklm12nopqrstuvwx","isDemo":1,"uid":12345678,"platform":2}]"""
#demo = True

ssid = """42["auth",{"session":"onhvo09okq5u7q8l8vhvricje4","isDemo":1,"uid":71098000,"platform":2}]"""
demo = True

min_payout = 92
period = 30
expiration = 30
api = PocketOption(ssid,demo)

# Lock to ensure only one trade at a time
trade_lock = threading.Lock()

# Global list to store trade logs
trade_logs = []

# Martingale step tracking per pair
martingale_steps = {}

# Max Martingale steps
max_mg_steps = 3

# Base trade amount
base_amount = 1

# Connect to API
api.connect()


def get_payout():
    try:
        from tabulate import tabulate
        d = global_value.PayoutData
        d = json.loads(d)
        table_data = []
        for pair in d:
            if len(pair) == 19:
                table_data.append([pair[1], pair[2], pair[3], pair[14], pair[5]])
                if pair[14] == True and pair[5] >= min_payout and "_otc" in pair[1]:
                    p = {}
                    p['id'] = pair[0]
                    p['payout'] = pair[5]
                    p['type'] = pair[3]
                    global_value.pairs[pair[1]] = p
        headers = ["ID", "Name", "Type", "Active", "Payout"]
        table_str = tabulate(table_data, headers, tablefmt="fancy_grid")
        global_value.logger(f"Payout Data:\n{table_str}", "DEBUG")
        return True
    except ImportError:
        # tabulate not installed, fallback to original logging
        for pair in d:
            if len(pair) == 19:
                global_value.logger('id: %s, name: %s, typ: %s, active: %s' % (str(pair[1]), str(pair[2]), str(pair[3]), str(pair[14])), "DEBUG")
        return True
    except:
        return False


def get_df():
    try:
        i = 0
        for pair in global_value.pairs:
            i += 1
            df = api.get_candles(pair, period)
            global_value.logger('%s (%s/%s)' % (str(pair), str(i), str(len(global_value.pairs))), "INFO")
            time.sleep(1)
        return True
    except:
        return False


def buy(amount, pair, action, expiration):
    global_value.logger('%s, %s, %s, %s' % (str(amount), str(pair), str(action), str(expiration)), "INFO")
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    i = result[1]
    result = api.check_win(i)
    if result:
        global_value.logger(str(result), "INFO")


def buy2(amount, pair, action, expiration, mg_step=0):
    with trade_lock:
        from tabulate import tabulate
        # ANSI color codes
        COLOR_RESET = "\033[0m"
        COLOR_GREEN = "\033[92m"
        COLOR_RED = "\033[91m"
        COLOR_YELLOW = "\033[93m"
        result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
        trade_id = result[1]
        # Check the result of the trade in real-time
        win_result = api.check_win(trade_id)
        if win_result is not None:
            win_value = win_result[0] if isinstance(win_result, tuple) else win_result
            result_str = "WIN" if win_value > 0 else "LOSS"

            # Update martingale steps based on result
            if pair not in martingale_steps:
                martingale_steps[pair] = 0

            if result_str == "WIN":
                martingale_steps[pair] = 0
            else:
                if mg_step < max_mg_steps:
                    martingale_steps[pair] = mg_step + 1
                else:
                    martingale_steps[pair] = 0

            # Append trade log with MG step
            trade_logs.append([amount, pair, action, expiration, result_str, mg_step])

            # Color the result column
            colored_trade_logs = []
            for log in trade_logs:
                colored_result = (COLOR_GREEN + log[4] + COLOR_RESET) if log[4] == "WIN" else (COLOR_RED + log[4] + COLOR_RESET)
                colored_trade_logs.append([log[0], log[1], log[2], log[3], colored_result, log[5]])
            # Prepare table and summary
            headers = ["Amount", "Pair", "Action", "Expiration", "Result", "MG"]
            table_str = tabulate(colored_trade_logs, headers, tablefmt="fancy_grid")
            next_analysis = wait(False)
            total_profit = sum([base_amount if log[4] == "WIN" else -base_amount for log in trade_logs])  # Assuming fixed base amount per trade

            # Calculate total trades and win rate
            total_trades = len(trade_logs)
            wins = sum(1 for log in trade_logs if log[4] == "WIN")
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

            summary = f"{COLOR_YELLOW}Next analysis in {next_analysis} seconds; Total profit: {total_profit}; Total trades: {total_trades}; Win rate: {win_rate:.2f}%{COLOR_RESET}"
            global_value.logger(f"Trade Log:\n{table_str}\n{summary}", "INFO")

            # If loss and mg_step < max, place next martingale trade automatically
            if result_str == "LOSS" and mg_step < max_mg_steps:
                next_amount = base_amount * (2 ** (mg_step + 1))
                # Fix martingale step direction: 1st step opposite to initial, 2nd opposite to 1st, 3rd opposite to 2nd, etc.
                next_action = "put" if action == "call" else "call"
                global_value.logger(f"Martingale step {mg_step + 1} for {pair}, placing next trade with amount {next_amount} with action {next_action}", "INFO")
                t = threading.Thread(target=buy2, args=(next_amount, pair, next_action, expiration, mg_step + 1))
                t.start()


def make_df(df0, history):
    df1 = pd.DataFrame(history).reset_index(drop=True)
    df1 = df1.sort_values(by='time').reset_index(drop=True)
    df1['time'] = pd.to_datetime(df1['time'], unit='s')
    df1.set_index('time', inplace=True)
    # df1.index = df1.index.floor('1s')

    df = df1['price'].resample(f'{period}s').ohlc()
    df.reset_index(inplace=True)
    df = df.loc[df['time'] < datetime.fromtimestamp(wait(False))]

    if df0 is not None:
        ts = datetime.timestamp(df.loc[0]['time'])
        for x in range(0, len(df0)):
            ts2 = datetime.timestamp(df0.loc[x]['time'])
            if ts2 < ts:
                df = df._append(df0.loc[x], ignore_index = True)
            else:
                break
        df = df.sort_values(by='time').reset_index(drop=True)
        df.set_index('time', inplace=True)
        df.reset_index(inplace=True)

    return df


def accelerator_oscillator(dataframe, fastPeriod=5, slowPeriod=34, smoothPeriod=5):
    ao = ta.SMA(dataframe["hl2"], timeperiod=fastPeriod) - ta.SMA(dataframe["hl2"], timeperiod=slowPeriod)
    ac = ta.SMA(ao, timeperiod=smoothPeriod)
    return ac


def DeMarker(dataframe, Period=14):
    dataframe['dem_high'] = dataframe['high'] - dataframe['high'].shift(1)
    dataframe['dem_low'] = dataframe['low'].shift(1) - dataframe['low']
    dataframe.loc[(dataframe['dem_high'] < 0), 'dem_high'] = 0
    dataframe.loc[(dataframe['dem_low'] < 0), 'dem_low'] = 0

    dem = ta.SMA(dataframe['dem_high'], Period) / (ta.SMA(dataframe['dem_high'], Period) + ta.SMA(dataframe['dem_low'], Period))
    return dem


def vortex_indicator(dataframe, Period=14):
    vm_plus = abs(dataframe['high'] - dataframe['low'].shift(1))
    vm_minus = abs(dataframe['low'] - dataframe['high'].shift(1))

    tr1 = dataframe['high'] - dataframe['low']
    tr2 = abs(dataframe['high'] - dataframe['close'].shift(1))
    tr3 = abs(dataframe['low'] - dataframe['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_vm_plus = vm_plus.rolling(window=Period).sum()
    sum_vm_minus = vm_minus.rolling(window=Period).sum()
    sum_tr = tr.rolling(window=Period).sum()

    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


def supertrend(df, multiplier, period):
    #df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = ta.SMA(df['TR'], period)

    st = 'ST'
    stx = 'STX'

    # Compute basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df


def strategie():
    for pair in global_value.pairs:
        if 'history' in global_value.pairs[pair]:
            history = []
            history.extend(global_value.pairs[pair]['history'])
            if 'dataframe' in global_value.pairs[pair]:
                df = make_df(global_value.pairs[pair]['dataframe'], history)
            else:
                df = make_df(None, history)
            
            # Skip new trades if Martingale sequence active for this pair
            if martingale_steps.get(pair, 0) > 0:
                continue
            
            # New condition: 1 bullish candle after bearish candle -> call
            if len(df) >= 2:
                last_two = df.iloc[-2:]
                first_candle = last_two.iloc[0]
                second_candle = last_two.iloc[1]
                first_bearish = first_candle['close'] < first_candle['open']
                second_bullish = second_candle['close'] > second_candle['open']
                first_bullish = first_candle['close'] > first_candle['open']
                second_bearish = second_candle['close'] < second_candle['open']

                # Calculate candle body sizes
                first_body_size = abs(first_candle['close'] - first_candle['open'])
                second_body_size = abs(second_candle['close'] - second_candle['open'])
                # Define minimum body size threshold (e.g., 0.1% of close price)
                min_body_size = second_candle['close'] * 0.001

                if first_bearish and second_bullish and second_body_size >= min_body_size:
                    if not trade_lock.locked():
                        action = "call"
                        global_value.logger(f"Placing trade: 1 bullish candle after bearish candle detected for {pair}, action: {action}", "INFO")
                        t = threading.Thread(target=buy2, args=(base_amount, pair, action, expiration, 0))
                        t.start()
                elif first_bullish and second_bearish and second_body_size >= min_body_size:
                    if not trade_lock.locked():
                        action = "put"
                        global_value.logger(f"Placing trade: 1 bearish candle after bullish candle detected for {pair}, action: {action}", "INFO")
                        t = threading.Thread(target=buy2, args=(base_amount, pair, action, expiration, 0))
                        t.start()

           

def prepare_get_history():
    try:
        data = get_payout()
        if data: return True
        else: return False
    except:
        return False

def prepare():
    try:
        data = get_payout()
        if data:
            data = get_df()
            if data: return True
            else: return False
        else: return False
    except:
        return False


def wait(sleep=True):
    dt = int(datetime.now().timestamp()) - int(datetime.now().second)
    if period == 60:
        dt += 120  # Increased sleep time to 2 minutes
    elif period == 120:
        dt += 180  # Increased sleep time to 2 minutes
    elif period == 30:
        if datetime.now().second < 30: dt += 30
        else: dt += 60
        if not sleep: dt -= 30
    elif period == 15:
        dt += 30  # Increased sleep time to 2 minutes for 15s candles
    elif period == 10:
        if datetime.now().second >= 50: dt += 60
        elif datetime.now().second >= 40: dt += 50
        elif datetime.now().second >= 30: dt += 40
        elif datetime.now().second >= 20: dt += 30
        elif datetime.now().second >= 10: dt += 20
        else: dt += 10
        if not sleep: dt -= 10
    elif period == 5:
        if datetime.now().second >= 55: dt += 60
        elif datetime.now().second >= 50: dt += 55
        elif datetime.now().second >= 45: dt += 50
        elif datetime.now().second >= 40: dt += 45
        elif datetime.now().second >= 35: dt += 40
        elif datetime.now().second >= 30: dt += 35
        elif datetime.now().second >= 25: dt += 30
        elif datetime.now().second >= 20: dt += 25
        elif datetime.now().second >= 15: dt += 20
        elif datetime.now().second >= 10: dt += 15
        elif datetime.now().second >= 5: dt += 10
        else: dt += 5
        if not sleep: dt -= 5
    elif period == 120:
        dt = int(datetime(int(datetime.now().year), int(datetime.now().month), int(datetime.now().day), int(datetime.now().hour), int(math.floor(int(datetime.now().minute) / 2) * 2), 0).timestamp())
        dt += 120
    elif period == 180:
        dt = int(datetime(int(datetime.now().year), int(datetime.now().month), int(datetime.now().day), int(datetime.now().hour), int(math.floor(int(datetime.now().minute) / 3) * 3), 0).timestamp())
        dt += 180
    elif period == 300:
        dt = int(datetime(int(datetime.now().year), int(datetime.now().month), int(datetime.now().day), int(datetime.now().hour), int(math.floor(int(datetime.now().minute) / 5) * 5), 0).timestamp())
        dt += 300
    elif period == 600:
        dt = int(datetime(int(datetime.now().year), int(datetime.now().month), int(datetime.now().day), int(datetime.now().hour), int(math.floor(int(datetime.now().minute) / 10) * 10), 0).timestamp())
        dt += 600
    if sleep:
        import sys
        for remaining in range(dt - int(datetime.now().timestamp()), 0, -1):
            sys.stdout.write(f'\r======== Sleeping {remaining} Seconds ========')
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write('\n')
        return 0

    return dt


def start():
    while global_value.websocket_is_connected is False:
        time.sleep(0.1)
    time.sleep(2)
    saldo = api.get_balance()
    try:
        from tabulate import tabulate
        table_str = tabulate([["Balance", saldo]], headers=["Account", "Balance"], tablefmt="fancy_grid")
        global_value.logger(f"Account Balance:\n{table_str}", "INFO")
    except ImportError:
        global_value.logger('Account Balance: %s' % str(saldo), "INFO")
    prep = prepare()
    if prep:
        while True:
            strategie()
            time.sleep(wait())


def start_get_history():
    while global_value.websocket_is_connected is False:
        time.sleep(0.1)
    time.sleep(2)
    saldo = api.get_balance()
    global_value.logger('Account Balance: %s' % str(saldo), "INFO")
    prep = prepare_get_history()
    if prep:
        i = 0
        for pair in global_value.pairs:
            i += 1
            global_value.logger('%s (%s/%s)' % (str(pair), str(i), str(len(global_value.pairs))), "INFO")
            if not global_value.check_cache(str(global_value.pairs[pair]["id"])):
                time_red = int(datetime.now().timestamp()) - 86400 * 7
                df = api.get_history(pair, period, end_time=time_red)


if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    rund = math.ceil(end_counter - start_counter)
    # print(f'CPU-gebundene Task-Zeit: {rund} {end_counter - start_counter} Sekunden')
    global_value.logger("CPU-gebundene Task-Zeit: %s Sekunden" % str(int(end_counter - start_counter)), "INFO")


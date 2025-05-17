import time, math, asyncio, json, threading
from datetime import datetime
import pocketoptionapi.global_value as global_value
import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text
from rich import box
import os
from pocketoptionapi.stable_api import PocketOption

# Initialize Rich console
console = Console()

# Set logging level
global_value.loglevel = 'INFO'

# Session configuration
start_counter = time.perf_counter()

# Clear terminal and show welcome message
os.system('cls' if os.name == 'nt' else 'clear')
console.print(Panel.fit(
    "[bold green]Opposite Martingale Trading Bot[/bold green]",
    border_style="green",
    subtitle="[italic]v1.0 - Binary Options Edition[/italic]"
))

# SSID configuration
ssid = """42["auth",{"session":"onhvo09okq5u7q8l8vhvricje4","isDemo":1,"uid":71098000,"platform":2}]"""
demo = True

min_payout = 92
period = 30
expiration = 30
api = PocketOption(ssid, demo)

# Lock to ensure only one trade at a time
trade_lock = threading.Lock()

# Global list to store trade logs
trade_logs = []

# Martingale step tracking per pair
martingale_steps = {}

# Max Martingale steps
max_mg_steps = 7

# Base trade amount
base_amount = 1

# Active trade flags
active_trade = False
active_martingale_pair = None

# Connect to API
api.connect()

# Custom logger using Rich with log level filtering
def custom_logger(message, level="INFO"):
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    current_level = levels.get(global_value.loglevel.upper(), 20)
    message_level = levels.get(level.upper(), 20)
    if message_level >= current_level:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_colors = {
            "INFO": "cyan",
            "DEBUG": "blue",
            "WARNING": "yellow",
            "ERROR": "red"
        }
        level_color = level_colors.get(level, "cyan")
        console.print(f"[{level_color}][{timestamp}] {level:8}[/{level_color}] | {message}")

# Override default logger
global_value.logger = custom_logger

def get_payout():
    try:
        d = global_value.PayoutData
        d = json.loads(d)
        
        # Create a Rich table for visualization
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Active", justify="center")
        table.add_column("Payout %", justify="right")
        
        available_pairs = []
        for pair in d:
            if len(pair) == 19:
                is_active = "✓" if pair[14] else "✗"
                active_style = "green" if pair[14] else "red"
                payout_style = "green" if pair[5] >= min_payout else "yellow"
                
                table.add_row(
                    str(pair[0]),
                    str(pair[1]),
                    str(pair[3]),
                    f"[{active_style}]{is_active}[/{active_style}]",
                    f"[{payout_style}]{pair[5]}%[/{payout_style}]"
                )
                
                if pair[14] == True and pair[5] >= min_payout and "_otc" in pair[1]:
                    p = {}
                    p['id'] = pair[0]
                    p['payout'] = pair[5]
                    p['type'] = pair[3]
                    global_value.pairs[pair[1]] = p
                    available_pairs.append(pair[1])
        
        console.print(Panel(table, title="[bold]Available Trading Pairs[/bold]", border_style="blue"))
        console.print(f"[bold green]Found {len(available_pairs)} valid trading pairs[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error retrieving payout data: {str(e)}[/bold red]")
        return False

def get_df():
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading candle data...", total=len(global_value.pairs))
            i = 0
            for pair in global_value.pairs:
                i += 1
                df = api.get_candles(pair, period)
                progress.update(task, advance=1, description=f"[cyan]Loading {pair}...")
                time.sleep(1)
        console.print("[bold green]✓ All candle data loaded successfully![/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error loading candle data: {str(e)}[/bold red]")
        return False

def buy(amount, pair, action, expiration):
    console.print(f"[cyan]Placing trade: Amount=${amount}, Pair={pair}, Action={action.upper()}, Expiration={expiration}s[/cyan]")
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    i = result[1]
    result = api.check_win(i)
    if result:
        console.print(f"[cyan]Trade result: {result}[/cyan]")

def buy2(amount, pair, action, expiration, mg_step=0):
    global active_trade, active_martingale_pair

    # Temporary storage for last losing martingale trade per pair
    if not hasattr(buy2, "last_loss_trade"):
        buy2.last_loss_trade = {}

    # Only check lock for new trades, not Martingale steps
    if mg_step == 0:
        with trade_lock:
            if active_trade and active_martingale_pair != pair:
                custom_logger(f"Trade already in progress, skipping {pair}", "WARNING")
                return
            active_trade = True
            active_martingale_pair = pair

    try:
        # Log exact trade entry time
        entry_time = datetime.now()
        candle_start = entry_time.replace(second=0, microsecond=0)
        delay = (entry_time - candle_start).total_seconds()

        # Only log timing for initial trades
        if mg_step == 0:
            console.print(f"[cyan]Trade placed at {entry_time.strftime('%H:%M:%S.%f')} "
                          f"(+{delay:.3f}s from candle open)[/cyan]")

        action_color = "green" if action == "call" else "red"
        console.print(Panel(
            f"[bold]Pair:[/bold] {pair}\n"
            f"[bold]Action:[/bold] [{action_color}]{action.upper()}[/{action_color}]\n"
            f"[bold]Amount:[/bold] ${amount:.2f}\n"
            f"[bold]Expiration:[/bold] {expiration}s\n"
            f"[bold]MG Step:[/bold] {mg_step}",
            title="[bold]PLACING TRADE[/bold]",
            border_style=action_color
        ))

        result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
        trade_id = result[1]

        # Show progress bar while waiting for trade result
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Waiting for trade result...", total=expiration)
            for i in range(expiration):
                time.sleep(1)
                progress.update(task, advance=1)

        win_result = api.check_win(trade_id)

        # Handle win result correctly
        if isinstance(win_result, dict):
            win_value = win_result.get('profit', 0)  # Get profit from dict
        else:
            win_value = win_result[0] if isinstance(win_result, tuple) else win_result

        result_str = "WIN" if win_value > 0 else "LOSS"
        profit = win_value if result_str == "WIN" else -amount

        # Update martingale steps
        if pair not in martingale_steps:
            martingale_steps[pair] = 0

        if result_str == "WIN":
            martingale_steps[pair] = 0
            console.print(f"[bold green]✓ TRADE WON: +${profit:.2f}[/bold green]")
            # On win, log immediately and clear any stored last loss trade
            timestamp = datetime.now().strftime("%H:%M:%S")
            trade_logs.append([timestamp, amount, pair, action, expiration, result_str, mg_step, profit])
            if pair in buy2.last_loss_trade:
                del buy2.last_loss_trade[pair]
        else:
            if mg_step < max_mg_steps:
                martingale_steps[pair] = mg_step + 1
                # Store last losing trade info but do not log yet
                timestamp = datetime.now().strftime("%H:%M:%S")
                buy2.last_loss_trade[pair] = [timestamp, amount, pair, action, expiration, result_str, mg_step, profit]
            else:
                martingale_steps[pair] = 0
                console.print(f"[bold red]✗ TRADE LOST: -${abs(profit):.2f}[/bold red]")
                # On final loss, log the stored last loss trade or current if none stored
                if pair in buy2.last_loss_trade:
                    trade_logs.append(buy2.last_loss_trade[pair])
                    del buy2.last_loss_trade[pair]
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    trade_logs.append([timestamp, amount, pair, action, expiration, result_str, mg_step, profit])

        # Display trade log table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Time")
        table.add_column("Amount")
        table.add_column("Pair")
        table.add_column("Action")
        table.add_column("Expiration")
        table.add_column("Result")
        table.add_column("P/L")
        table.add_column("MG Step")

        for log in trade_logs[-1000:]:  # Show last 100 trades
            timestamp, amount, pair, action, exp, result, mg, profit = log
            result_style = "green" if result == "WIN" else "red"
            profit_style = "green" if profit >= 0 else "red"

            table.add_row(
                timestamp,
                f"${amount:.2f}",
                pair,
                action.upper(),
                f"{exp}s",
                f"[{result_style}]{result}[/{result_style}]",
                f"[{profit_style}]${profit:.2f}[/{profit_style}]",
                str(mg)
            )

        # Calculate summary stats
        next_analysis = wait(period, False)
        total_profit = sum(log[7] for log in trade_logs)
        total_trades = len(trade_logs)
        wins = sum(1 for log in trade_logs if log[5] == "WIN")
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # Calculate real-time remaining seconds until next analysis
        remaining_seconds = int(next_analysis - datetime.now().timestamp())

        console.print(Panel(
            table,
            title="[bold]Recent Trades[/bold]",
            border_style="blue"
        ))
        console.print(
            f"[yellow]Next analysis in {remaining_seconds}s | "
            f"Total Profit: ${total_profit:.2f} | "
            f"Trades: {total_trades} | "
            f"Win Rate: {win_rate:.2f}%[/yellow]"
        )

        # Modify the Martingale handling section:
        if result_str == "LOSS" and mg_step < max_mg_steps:
            next_amount = base_amount * (2 ** (mg_step + 1))
            next_action = "put" if action == "call" else "call"
            console.print(f"[bold yellow]Starting Martingale step {mg_step + 1} for {pair}[/bold yellow]")

            # Add small delay before next Martingale trade
            time.sleep(0.005)

            # Execute next Martingale trade directly
            buy2(next_amount, pair, next_action, expiration, mg_step + 1)
        else:
            # Reset flags only when Martingale sequence is complete
            active_trade = False
            active_martingale_pair = None
            martingale_steps[pair] = 0

    except Exception as e:
        console.print(f"[bold red]Error in trade execution: {str(e)}[/bold red]")
        active_trade = False
        active_martingale_pair = None
        martingale_steps[pair] = 0

def make_df(df0, history):
    df1 = pd.DataFrame(history).reset_index(drop=True)
    df1 = df1.sort_values(by='time').reset_index(drop=True)
    df1['time'] = pd.to_datetime(df1['time'], unit='s')
    df1.set_index('time', inplace=True)
    df = df1['price'].resample(f'{period}s').ohlc()
    df.reset_index(inplace=True)
    df = df.loc[df['time'] < datetime.fromtimestamp(wait(period, False))]
    
    if df0 is not None:
        ts = datetime.timestamp(df.loc[0]['time'])
        for x in range(0, len(df0)):
            ts2 = datetime.timestamp(df0.loc[x]['time'])
            if ts2 < ts:
                df = df._append(df0.loc[x], ignore_index=True)
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
    df['TR'] = ta.TRANGE(df)
    df['ATR'] = ta.SMA(df['TR'], period)
    st = 'ST'
    stx = 'STX'
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i] else 0.00
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    df.fillna(0, inplace=True)
    return df

def strategie():
    global active_trade, active_martingale_pair
    
    # If there's an active trade or Martingale sequence, skip analysis
    if active_trade:
        return
        
    # Check if we're close to candle opening
    current_time = datetime.now()
    seconds_into_candle = current_time.second % period
    milliseconds = current_time.microsecond / 1000000
    total_seconds_offset = seconds_into_candle + milliseconds
    
    # Only proceed if we're within 1 second of candle opening
    if total_seconds_offset > 1:
        return  # Skip analysis entirely if not near candle opening
        
    for pair in global_value.pairs:
        # Skip if this pair has an active Martingale sequence
        if martingale_steps.get(pair, 0) > 0:
            continue
            
        if 'history' in global_value.pairs[pair]:
            history = []
            history.extend(global_value.pairs[pair]['history'])
            if 'dataframe' in global_value.pairs[pair]:
                df = make_df(global_value.pairs[pair]['dataframe'], history)
            else:
                df = make_df(None, history)
            
            # Skip new trades if Martingale sequence active
            if martingale_steps.get(pair, 0) > 0:
                continue
            
            # Trading condition: 1 bullish candle after bearish candle -> call
            if len(df) >= 2:
                last_two = df.iloc[-2:]
                first_candle = last_two.iloc[0]
                second_candle = last_two.iloc[1]
                first_bearish = first_candle['close'] < first_candle['open']
                second_bullish = second_candle['close'] > second_candle['open']
                first_bullish = first_candle['close'] > first_candle['open']
                second_bearish = second_candle['close'] < second_candle['open']
                first_body_size = abs(first_candle['close'] - first_candle['open'])
                second_body_size = abs(second_candle['close'] - second_candle['open'])
                min_body_size = second_candle['close'] * 0.001
                
                if first_bearish and second_bullish and second_body_size >= min_body_size:
                    signal_detected = True
                    action = "call"
                elif first_bullish and second_bearish and second_body_size >= min_body_size:
                    signal_detected = True
                    action = "put"
                else:
                    signal_detected = False
                
                if signal_detected:
                    next_candle = wait(period, False)
                    current_time = datetime.now().timestamp()
                    
                    if next_candle - current_time <= 2:
                        # Double-check active trade status before placing trade
                        if not active_trade:
                            buy2(base_amount, pair, action, expiration)
                        break  # Exit loop after placing trade

def prepare_get_history():
    try:
        data = get_payout()
        if data:
            console.print("[bold green]Payout data retrieved successfully[/bold green]")
            return True
        else:
            console.print("[bold red]Failed to retrieve payout data[/bold red]")
            return False
    except Exception as e:
        console.print(f"[bold red]Error in prepare_get_history: {str(e)}[/bold red]")
        return False

def prepare():
    try:
        data = get_payout()
        if data:
            data = get_df()
            if data:
                console.print("[bold green]Initialization complete[/bold green]")
                return True
            else:
                console.print("[bold red]Failed to retrieve candlestick data[/bold red]")
                return False
        else:
            console.print("[bold red]Failed to retrieve payout data[/bold red]")
            return False
    except Exception as e:
        console.print(f"[bold red]Error in prepare: {str(e)}[/bold red]")
        return False

def wait(period, sleep=True):
    try:
        current_time = datetime.now()
        # Calculate next candle open time
        dt = int(current_time.timestamp()) - int(current_time.second) - int(current_time.microsecond/1000000)
        
        # Align to the exact next period start
        if period in [5, 10, 15, 30, 60]:
            seconds_until_next = period - (current_time.second % period)
            dt += seconds_until_next
        elif period in [120, 180, 300, 600]:
            minutes_until_next = period/60 - (current_time.minute % (period/60))
            dt += minutes_until_next * 60
            
        if not sleep:
            return dt
            
        # Wait until exactly candle open
        remaining = dt - current_time.timestamp()
        while datetime.now().timestamp() < dt:
            time.sleep(0.001)  # 1ms intervals for precise timing
            
        return 0
    except Exception as e:
        console.print(f"[bold red]Error in wait function: {str(e)}[/bold red]")
        return 0

def start():
    while global_value.websocket_is_connected is False:
        time.sleep(0.1)
    time.sleep(2)
    saldo = api.get_balance()
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Account", style="cyan")
    table.add_column("Balance", style="cyan")
    table.add_row("Demo" if demo else "Real", f"${saldo:.2f}")
    console.print(Panel(table, title="[bold]Account Balance[/bold]", border_style="blue"))
    
    prep = prepare()
    if prep:
        while True:
            strategie()
            time.sleep(wait(period))
    else:
        console.print("[bold red]Initialization failed, exiting[/bold red]")

def start_get_history():
    while global_value.websocket_is_connected is False:
        time.sleep(0.1)
    time.sleep(2)
    saldo = api.get_balance()
    console.print(f"[cyan]Account Balance: ${saldo:.2f}[/cyan]")
    prep = prepare_get_history()
    if prep:
        i = 0
        for pair in global_value.pairs:
            i += 1
            console.print(f"[cyan]Fetching history for {pair} ({i}/{len(global_value.pairs)})[/cyan]")
            if not global_value.check_cache(str(global_value.pairs[pair]["id"])):
                time_red = int(datetime.now().timestamp()) - 86400 * 7
                df = api.get_history(pair, period, end_time=time_red)

if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    rund = math.ceil(end_counter - start_counter)
    console.print(f"[bold green]Execution Time: {rund} seconds[/bold green]")
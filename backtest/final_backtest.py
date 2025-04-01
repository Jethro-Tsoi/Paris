import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
import ast
import time
import re

# Map for tickers that need special treatment for Yahoo Finance
TICKER_REPLACEMENTS = {
    'SPX': '^SPX',    # S&P 500 Index
    'DXY': 'DX-Y.NYB', # US Dollar Index
    'NDX': '^NDX',    # NASDAQ-100 Index
    'VIX': '^VIX',    # Volatility Index
    'BTC': 'BTC-USD', # Bitcoin
    'ES': 'ES=F',     # E-mini S&P 500 Futures
    'CL': 'CL=F',     # Crude Oil Futures
    'GC': 'GC=F',     # Gold Futures
    'ZB': 'ZB=F',     # Treasury Bond Futures
    'NQ': 'NQ=F',     # Nasdaq 100 Futures
    'RTY': 'RTY=F',   # Russell 2000 Futures
    'ZN': 'ZN=F',     # 10-Year T-Note Futures
    'EUR': 'EURUSD=X', # Euro to USD
    'JPY': 'JPY=X'    # Japanese Yen
}

def load_predictions(file_path):
    """
    Load the FinBERT predictions CSV file
    """
    print(f"Loading predictions from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Print some examples of timestamp format
    if 'timestamp' in df.columns:
        print("\nTimestamp format examples:")
        for ts in df['timestamp'].head(5):
            print(f"  {ts}")
    
    return df

def extract_tickers(df):
    """
    Extract and group by tickers from the financial_info column
    which contains JSON-like data
    """
    print("Extracting tickers from financial_info column...")
    ticker_groups = defaultdict(list)
    error_count = 0
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('financial_info')) and row.get('financial_info'):
            try:
                # The financial_info column contains string representations of lists of dictionaries
                # Convert it to Python objects using ast.literal_eval
                financial_info = ast.literal_eval(row['financial_info'])
                
                # Process each ticker in the financial_info
                for info in financial_info:
                    if 'ticker' in info and info['ticker'].startswith('$'):
                        ticker = info['ticker'].strip('$')
                        if ticker and len(ticker) <= 5 and ticker.isalpha():
                            ticker_groups[ticker].append({
                                'idx': idx,
                                'timestamp': row.get('timestamp'),
                                'sentiment': row.get('finbert_predicted_sentiment'),
                                'price': info.get('price'),
                                'percentage_change': info.get('percentage_change')
                            })
            except (ValueError, SyntaxError, TypeError) as e:
                # Skip rows with invalid financial_info format
                error_count += 1
                if error_count <= 10 or idx % 1000 == 0:  # Limit error reporting
                    print(f"Error parsing financial_info at row {idx}: {str(e)}")
                continue
    
    print(f"Found {len(ticker_groups)} unique tickers")
    if error_count > 0:
        print(f"Encountered {error_count} errors during ticker extraction")
    return ticker_groups

def get_stock_data(ticker, start_date, end_date, retry_count=3, retry_delay=2):
    """
    Get historical stock data using yfinance with rate limiting
    """
    # Check if ticker needs special treatment
    if ticker in TICKER_REPLACEMENTS:
        yahoo_ticker = TICKER_REPLACEMENTS[ticker]
    else:
        yahoo_ticker = ticker
    
    for attempt in range(retry_count):
        try:
            print(f"Fetching data for {ticker} ({yahoo_ticker}) from {start_date} to {end_date}")
            stock = yf.Ticker(yahoo_ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data available for {ticker} ({yahoo_ticker})")
                return None
                
            print(f"Fetched {len(data)} days of data for {ticker}")
            # Add a delay to avoid rate limiting
            time.sleep(0.5)
            return data
        except Exception as e:
            print(f"Error retrieving data for {ticker} ({yahoo_ticker}): {str(e)}")
            if "Rate limit" in str(e) and attempt < retry_count - 1:
                sleep_time = retry_delay * (attempt + 1)
                print(f"Rate limited. Waiting {sleep_time}s before retry {attempt+2}/{retry_count}...")
                time.sleep(sleep_time)
            elif attempt < retry_count - 1:
                print(f"Retrying ({attempt+2}/{retry_count})...")
                time.sleep(retry_delay)
            else:
                return None

def analyze_ticker_sentiments(df, ticker_groups):
    """
    Analyze sentiments for each ticker
    """
    sentiment_stats = {}
    
    for ticker, mentions in ticker_groups.items():
        # Extract indices of mentions
        indices = [mention['idx'] for mention in mentions]
        ticker_df = df.loc[indices]
        
        # Calculate sentiment statistics
        sentiment_counts = ticker_df['finbert_predicted_sentiment'].value_counts()
        sentiment_percentages = sentiment_counts / len(ticker_df) * 100
        
        # Average sentiment scores
        avg_scores = {
            'STRONGLY_POSITIVE': ticker_df['finbert_score_STRONGLY_POSITIVE'].mean(),
            'POSITIVE': ticker_df['finbert_score_POSITIVE'].mean(),
            'NEUTRAL': ticker_df['finbert_score_NEUTRAL'].mean(), 
            'NEGATIVE': ticker_df['finbert_score_NEGATIVE'].mean(),
            'STRONGLY_NEGATIVE': ticker_df['finbert_score_STRONGLY_NEGATIVE'].mean()
        }
        
        # Store the statistics
        sentiment_stats[ticker] = {
            'count': len(ticker_df),
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'avg_scores': avg_scores,
            'most_common_sentiment': sentiment_counts.idxmax() if not sentiment_counts.empty else None
        }
    
    return sentiment_stats

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to datetime object
    """
    if pd.isna(timestamp_str) or not timestamp_str:
        return None
    
    try:
        # Handle ISO format with timezone: 2022-10-11T16:00:05.958000+00:00
        return pd.to_datetime(timestamp_str)
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {str(e)}")
        return None

def determine_date_range(df):
    """
    Determine the date range for backtesting based on available timestamps
    """
    if 'timestamp' not in df.columns:
        print("No timestamp column found, using default date range")
        return "2025-03-01", "2025-04-01"
    
    # Try to parse timestamps
    valid_dates = []
    
    for ts in df['timestamp'].dropna().unique():
        try:
            parsed_ts = pd.to_datetime(ts)
            valid_dates.append(parsed_ts.date())
        except:
            continue
    
    if len(valid_dates) < 10:
        print(f"Too few valid timestamps ({len(valid_dates)}), using default date range")
        return "2025-03-01", "2025-04-01"
    
    # Get min and max dates
    min_date = min(valid_dates)
    max_date = max(valid_dates)
    
    # Add a few days on either end
    start_date = (min_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = (max_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    print(f"Using date range based on timestamps: {start_date} to {end_date}")
    return start_date, end_date

def check_ticker_availability(tickers, max_tickers=30):
    """
    Check which tickers are available on Yahoo Finance
    Returns a list of available tickers
    """
    print(f"Checking availability of top {max_tickers} tickers...")
    
    available_tickers = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    count = 0
    for ticker in tickers:
        if count >= max_tickers:
            break
            
        count += 1
        if count % 5 == 0:
            print(f"Processed {count}/{max_tickers} tickers...")
            
        # Try to get a small amount of data
        stock_data = get_stock_data(ticker, one_month_ago, current_date, retry_count=2)
        
        if stock_data is not None and not stock_data.empty:
            available_tickers.append(ticker)
            print(f"✓ {ticker} is available")
        else:
            print(f"✗ {ticker} is not available")
            
        # Add delay to avoid rate limiting
        time.sleep(1.5)
    
    print(f"Found {len(available_tickers)} available tickers out of {count} checked")
    return available_tickers

def backtest_strategy(df, ticker_groups, days_to_hold=3, max_tickers=20):
    """
    Backtest a simple strategy where we buy/sell based on individual sentiment predictions
    and hold for n days
    """
    results = {}
    
    # Try to determine date range from timestamps, or use default
    start_date, end_date = determine_date_range(df)
    print(f"Backtesting from {start_date} to {end_date}")
    
    # Get top tickers by mention count
    sorted_tickers = sorted(ticker_groups.keys(), 
                           key=lambda x: len(ticker_groups[x]), 
                           reverse=True)[:max_tickers]
    
    print(f"Testing top {len(sorted_tickers)} tickers: {', '.join(sorted_tickers)}")
    
    # Process each ticker directly with full historical data
    available_tickers = []
    for ticker in sorted_tickers:
        mentions = ticker_groups[ticker]
        print(f"\nAnalyzing {ticker} with {len(mentions)} mentions...")
        
        # Get full historical stock data for the entire period
        stock_data = get_stock_data(ticker, start_date, end_date, retry_count=3, retry_delay=3)
        if stock_data is None or stock_data.empty:
            print(f"No data available for {ticker}, skipping...")
            continue
            
        print(f"Fetched {len(stock_data)} days of data for {ticker} from {start_date} to {end_date}")
        
        # Calculate some basic statistics for the full period
        ticker_stats = {}
        if len(stock_data) > 1:
            first_price = stock_data.iloc[0]['Close']
            last_price = stock_data.iloc[-1]['Close']
            period_return = ((last_price / first_price) - 1) * 100
            max_price = stock_data['High'].max()
            min_price = stock_data['Low'].min()
            max_drawdown = ((stock_data['Close'].cummax() - stock_data['Close']) / stock_data['Close'].cummax()).max() * 100
            daily_returns = stock_data['Close'].pct_change() * 100
            mean_return = daily_returns.mean()
            var_return = daily_returns.var()
            
            # Store stats in dictionary for later use
            ticker_stats = {
                'start_date': stock_data.index[0],
                'end_date': stock_data.index[-1],
                'first_price': first_price,
                'last_price': last_price,
                'period_return': period_return,
                'max_price': max_price,
                'min_price': min_price,
                'max_drawdown': max_drawdown,
                'mean_return': mean_return,
                'var_return': var_return
            }
            
            print(f"Full period stats for {ticker}:")
            print(f"  - Date range: {ticker_stats['start_date']} to {ticker_stats['end_date']}")
            print(f"  - Starting price: ${ticker_stats['first_price']:.2f}")
            print(f"  - Ending price: ${ticker_stats['last_price']:.2f}")
            print(f"  - Period return: {ticker_stats['period_return']:.2f}%")
            print(f"  - Highest price: ${ticker_stats['max_price']:.2f}")
            print(f"  - Lowest price: ${ticker_stats['min_price']:.2f}")
            print(f"  - Max drawdown: {ticker_stats['max_drawdown']:.2f}%")
            print(f"  - Average daily return: {ticker_stats['mean_return']:.2f}%")
            print(f"  - Daily return variance: {ticker_stats['var_return']:.2f}%")
        
        available_tickers.append(ticker)
        
        # Prepare for tracking trades
        trades = []
        
        # Iterate through each mention/signal for this ticker
        print(f"Processing {len(mentions)} individual signals for {ticker}...")
        signal_count = 0
        neutral_count = 0
        invalid_count = 0
        
        for mention in mentions:
            # Skip neutral signals
            sentiment = mention['sentiment']
            if sentiment == 'NEUTRAL' or pd.isna(sentiment):
                neutral_count += 1
                continue
                
            # Determine if it's a buy or sell signal
            is_buy = sentiment in ['POSITIVE', 'STRONGLY_POSITIVE']
            
            # Parse the timestamp to get the entry date
            try:
                entry_date = parse_timestamp(mention['timestamp'])
                if entry_date is None:
                    invalid_count += 1
                    continue
                    
                entry_date_str = entry_date.strftime('%Y-%m-%d')
            except Exception as e:
                invalid_count += 1
                continue
            
            # Find the next trading day
            future_dates = stock_data.index[stock_data.index >= entry_date_str]
            if len(future_dates) == 0:
                invalid_count += 1
                continue
            
            entry_date = future_dates[0]
            
            # Calculate exit date (entry + holding period)
            exit_date = entry_date + pd.Timedelta(days=days_to_hold)
            
            # Find the actual exit date (nearest trading day)
            future_dates = stock_data.index[stock_data.index >= exit_date]
            if len(future_dates) == 0:
                if len(stock_data.index) > 0:
                    exit_date = stock_data.index[-1]
                else:
                    invalid_count += 1
                    continue
            else:
                exit_date = future_dates[0]
            
            # Get entry and exit prices
            try:
                entry_price = stock_data.loc[entry_date]['Close']
                exit_price = stock_data.loc[exit_date]['Close']
            except:
                invalid_count += 1
                continue
            
            # Calculate return (positive for buys if price went up, positive for sells if price went down)
            if is_buy:
                returns_pct = (exit_price / entry_price - 1) * 100
            else:
                returns_pct = (1 - exit_price / entry_price) * 100
                
            # Store the trade details
            trades.append({
                'ticker': ticker,
                'signal': 'BUY' if is_buy else 'SELL',
                'sentiment': sentiment,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'returns_pct': returns_pct
            })
            
            signal_count += 1
        
        print(f"Completed processing signals for {ticker}:")
        print(f"  - Valid signals processed: {signal_count}")
        print(f"  - Neutral signals skipped: {neutral_count}")
        print(f"  - Invalid signals skipped: {invalid_count}")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate trade performance statistics
            trade_returns = trades_df['returns_pct']
            
            # Calculate statistics
            results[ticker] = {
                'trades': trades_df,
                'total_trades': len(trades_df),
                'avg_return': trade_returns.mean(),
                'median_return': trade_returns.median(),
                'var_return': trade_returns.var(),
                'win_rate': (trade_returns > 0).mean() * 100,
                'max_gain': trade_returns.max(),
                'max_loss': trade_returns.min(),
                'total_return': trade_returns.sum(),
                'sentiment_counts': trades_df['sentiment'].value_counts().to_dict(),
            }
            
            # Add full period stats to results
            results[ticker].update(ticker_stats)
            
            # Show sentiment breakdown
            print(f"Sentiment breakdown for {ticker} trades:")
            for sentiment, count in results[ticker]['sentiment_counts'].items():
                print(f"  - {sentiment}: {count} trades")
            
            # Show trade performance
            print(f"Trade performance for {ticker}:")
            print(f"  - Total trades: {results[ticker]['total_trades']}")
            print(f"  - Average return: {results[ticker]['avg_return']:.2f}%")
            print(f"  - Median return: {results[ticker]['median_return']:.2f}%")
            print(f"  - Variance: {results[ticker]['var_return']:.2f}%")
            print(f"  - Win rate: {results[ticker]['win_rate']:.2f}%")
            print(f"  - Max gain: {results[ticker]['max_gain']:.2f}%")
            print(f"  - Max loss: {results[ticker]['max_loss']:.2f}%")
            print(f"  - Total return from all trades: {results[ticker]['total_return']:.2f}%")
        else:
            print(f"No valid trades for {ticker}")
    
    print(f"\nSuccessfully processed {len(available_tickers)} tickers: {', '.join(available_tickers)}")
    return results

def plot_backtest_results(results):
    """
    Plot the results of the backtest
    """
    if not results:
        print("No results to plot")
        return
    
    # Create output directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    tickers = list(results.keys())
    returns = [results[ticker]['avg_return'] for ticker in tickers]
    win_rates = [results[ticker]['win_rate'] for ticker in tickers]
    
    # 1. Plot average returns and win rates per ticker
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort tickers by returns
    sorted_idx = np.argsort(returns)
    sorted_tickers = [tickers[i] for i in sorted_idx]
    sorted_returns = [returns[i] for i in sorted_idx]
    sorted_win_rates = [win_rates[i] for i in sorted_idx]
    
    # Plot average returns
    bars1 = ax1.bar(sorted_tickers, sorted_returns)
    ax1.set_title('Average Return per Trade (%)')
    ax1.set_ylabel('Return (%)')
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_xticklabels(sorted_tickers, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Color bars by positive/negative returns
    for i, bar in enumerate(bars1):
        if sorted_returns[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Plot win rates
    bars2 = ax2.bar(sorted_tickers, sorted_win_rates)
    ax2.set_title('Win Rate (%)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.axhline(y=50, color='r', linestyle='-')
    ax2.set_xticklabels(sorted_tickers, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Color bars by win rate above/below 50%
    for i, bar in enumerate(bars2):
        if sorted_win_rates[i] >= 50:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/backtest_results_by_ticker.png')
    print(f"Plot saved to {output_dir}/backtest_results_by_ticker.png")
    plt.close()
    
    # 2. Create individual ticker analysis charts
    print("\nCreating individual ticker analysis charts...")
    for ticker in results:
        result = results[ticker]
        if 'period_return' not in result or result['period_return'] is None:
            continue
            
        # Create figure with 2 rows for full period stats and trade performance
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Full period stats
        full_period_metrics = [
            ('Period Return', result.get('period_return', 0)),
            ('Max Drawdown', result.get('max_drawdown', 0)),
            ('Avg Daily Return', result.get('mean_return', 0)),
            ('Variance', result.get('var_return', 0))
        ]
        
        metric_names = [m[0] for m in full_period_metrics]
        metric_values = [m[1] for m in full_period_metrics]
        
        # Plot full period stats
        bars1 = ax1.bar(metric_names, metric_values)
        ax1.set_title(f'{ticker} - Full Period Stats')
        ax1.set_ylabel('Value (%)')
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars1):
            if metric_values[i] >= 0:
                bar.set_color('blue')
            else:
                bar.set_color('purple')
                
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            ax1.text(i, v + 0.1 if v >= 0 else v - 2, f'{v:.2f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        # Trade performance metrics
        trade_metrics = [
            ('Avg Return', result['avg_return']),
            ('Median Return', result['median_return']),
            ('Win Rate', result['win_rate']),
            ('Max Gain', result['max_gain']),
            ('Max Loss', result['max_loss']),
            ('Total Return', result['total_return'])
        ]
        
        metric_names = [m[0] for m in trade_metrics]
        metric_values = [m[1] for m in trade_metrics]
        
        # Plot trade performance metrics
        bars2 = ax2.bar(metric_names, metric_values)
        ax2.set_title(f'{ticker} - Trade Performance ({result["total_trades"]} Trades)')
        ax2.set_ylabel('Value (%)')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars2):
            if metric_values[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            ax2.text(i, v + 0.1 if v >= 0 else v - 2, f'{v:.2f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        # Add text with additional details at the top of the figure
        plt.figtext(0.5, 0.95, f"{ticker} Analysis", ha='center', va='center', fontsize=14, fontweight='bold')
        
        trade_counts = result.get('sentiment_counts', {})
        sentiment_text = ", ".join([f"{sentiment}: {count}" for sentiment, count in trade_counts.items()])
        
        plt.figtext(0.5, 0.92, f"Trades by sentiment: {sentiment_text}", ha='center', va='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(f'{output_dir}/ticker_analysis_{ticker}.png')
        plt.close()
    
    # 3. Collect all trades to analyze by sentiment
    all_trades = []
    for ticker in results:
        if 'trades' in results[ticker]:
            ticker_trades = results[ticker]['trades']
            all_trades.append(ticker_trades)
    
    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Calculate overall performance metrics
        overall_results = {
            'total_trades': len(all_trades_df),
            'avg_return': all_trades_df['returns_pct'].mean(),
            'median_return': all_trades_df['returns_pct'].median(),
            'win_rate': (all_trades_df['returns_pct'] > 0).mean() * 100,
            'max_gain': all_trades_df['returns_pct'].max(),
            'max_loss': all_trades_df['returns_pct'].min(),
            'total_return': all_trades_df['returns_pct'].sum(),
            'sentiment_counts': all_trades_df['sentiment'].value_counts().to_dict()
        }
        
        # Display overall performance
        print("\nOverall Trade Performance (All Tickers):")
        print(f"  - Total trades: {overall_results['total_trades']}")
        print(f"  - Average return: {overall_results['avg_return']:.2f}%")
        print(f"  - Median return: {overall_results['median_return']:.2f}%")
        print(f"  - Win rate: {overall_results['win_rate']:.2f}%")
        print(f"  - Max gain: {overall_results['max_gain']:.2f}%")
        print(f"  - Max loss: {overall_results['max_loss']:.2f}%")
        print(f"  - Total return from all trades: {overall_results['total_return']:.2f}%")
        
        # Plot overall performance metrics
        plt.figure(figsize=(12, 8))
        
        overall_metrics = [
            ('Avg Return', overall_results['avg_return']),
            ('Median Return', overall_results['median_return']),
            ('Win Rate', overall_results['win_rate']),
            ('Max Gain', overall_results['max_gain']),
            ('Max Loss', overall_results['max_loss'])
        ]
        
        metric_names = [m[0] for m in overall_metrics]
        metric_values = [m[1] for m in overall_metrics]
        
        bars = plt.bar(metric_names, metric_values)
        
        # Color bars
        for i, bar in enumerate(bars):
            if i == 4:  # Max Loss is always negative
                bar.set_color('red')
            elif metric_values[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.1 if v >= 0 else v - 2, f'{v:.2f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.title(f'Overall Trade Performance ({overall_results["total_trades"]} Trades)')
        plt.ylabel('Value (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/overall_performance.png')
        print(f"Plot saved to {output_dir}/overall_performance.png")
        plt.close()
        
        # Group by sentiment and calculate detailed performance metrics
        sentiment_performance = {}
        for sentiment in all_trades_df['sentiment'].unique():
            sentiment_trades = all_trades_df[all_trades_df['sentiment'] == sentiment]
            sentiment_performance[sentiment] = {
                'total_trades': len(sentiment_trades),
                'avg_return': sentiment_trades['returns_pct'].mean(),
                'median_return': sentiment_trades['returns_pct'].median(),
                'win_rate': (sentiment_trades['returns_pct'] > 0).mean() * 100,
                'max_gain': sentiment_trades['returns_pct'].max(),
                'max_loss': sentiment_trades['returns_pct'].min(),
                'total_return': sentiment_trades['returns_pct'].sum()
            }
            
        # Display sentiment-based performance
        print("\nPerformance by Sentiment:")
        for sentiment, stats in sentiment_performance.items():
            print(f"\n{sentiment} ({stats['total_trades']} trades):")
            print(f"  - Average return: {stats['avg_return']:.2f}%")
            print(f"  - Median return: {stats['median_return']:.2f}%")
            print(f"  - Win rate: {stats['win_rate']:.2f}%")
            print(f"  - Total return: {stats['total_return']:.2f}%")
        
        # Sort sentiments for consistent plotting
        sentiment_order = {
            'STRONGLY_POSITIVE': 0,
            'POSITIVE': 1,
            'NEGATIVE': 2,
            'STRONGLY_NEGATIVE': 3
        }
        
        sentiments = list(sentiment_performance.keys())
        sentiments.sort(key=lambda x: sentiment_order.get(x, 99))
        
        # Plot key metrics by sentiment
        metrics_to_plot = ['avg_return', 'median_return', 'win_rate']
        colors = {
            'avg_return': 'blue',
            'median_return': 'green',
            'win_rate': 'orange'
        }
        
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            
            values = [sentiment_performance[s][metric] for s in sentiments]
            bars = plt.bar(sentiments, values)
            
            # Color bars based on sentiment
            sentiment_colors = {
                'STRONGLY_POSITIVE': 'darkgreen',
                'POSITIVE': 'green',
                'NEGATIVE': 'red',
                'STRONGLY_NEGATIVE': 'darkred'
            }
            
            for j, bar in enumerate(bars):
                bar.set_color(sentiment_colors.get(sentiments[j], 'blue'))
                
            # Add value labels
            for j, v in enumerate(values):
                plt.text(j, v + 0.1 if v >= 0 else v - 2, f'{v:.2f}%', 
                        ha='center', va='bottom' if v >= 0 else 'top')
            
            metric_display_name = metric.replace('_', ' ').title()
            plt.title(f'{metric_display_name} by Sentiment')
            plt.ylabel('Value (%)')
            
            if metric != 'win_rate':
                plt.axhline(y=0, color='r', linestyle='-')
            else:
                plt.axhline(y=50, color='r', linestyle='-')
                
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_by_sentiment.png')
        print(f"Plot saved to {output_dir}/performance_by_sentiment.png")
        plt.close()
        
        # Group by sentiment and calculate average returns
        sentiment_groups = all_trades_df.groupby('sentiment')
        sentiment_stats = sentiment_groups.agg({
            'returns_pct': ['mean', 'count'],
            'ticker': 'count'
        })
        
        # Prepare data for plotting
        sentiments = list(sentiment_stats.index)
        returns = sentiment_stats['returns_pct']['mean'].values
        counts = sentiment_stats['ticker']['count'].values
        
        # Sort by sentiment type (strongly positive -> strongly negative)
        sorted_idx = sorted(range(len(sentiments)), key=lambda i: sentiment_order.get(sentiments[i], 99))
        sorted_sentiments = [sentiments[i] for i in sorted_idx]
        sorted_returns = [returns[i] for i in sorted_idx]
        sorted_counts = [counts[i] for i in sorted_idx]
        
        # Plot returns by sentiment
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(sorted_sentiments, sorted_returns)
        
        # Color the bars
        colors = {
            'STRONGLY_POSITIVE': 'darkgreen',
            'POSITIVE': 'green', 
            'NEGATIVE': 'red',
            'STRONGLY_NEGATIVE': 'darkred'
        }
        
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(sorted_sentiments[i], 'blue'))
        
        # Add trade count as text on top of bars
        for i, (val, count) in enumerate(zip(sorted_returns, sorted_counts)):
            plt.text(i, val + 0.1, f'{count} trades', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.title('Average Return by Sentiment Signal (%)')
        plt.ylabel('Return (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/backtest_results_by_sentiment.png')
        print(f"Plot saved to {output_dir}/backtest_results_by_sentiment.png")
        plt.close()
        
        # Also plot the count of trades by sentiment
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_sentiments, sorted_counts)
        
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(sorted_sentiments[i], 'blue'))
            
        plt.title('Number of Trades by Sentiment')
        plt.ylabel('Number of Trades')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trade_count_by_sentiment.png')
        print(f"Plot saved to {output_dir}/trade_count_by_sentiment.png")
        plt.close()
    
    # Save the detailed results
    summary_data = []
    for ticker in results:
        result = results[ticker]
        
        summary_item = {
            'ticker': ticker,
            'sentiment_breakdown': str(result.get('sentiment_counts', {})),
            'total_trades': result['total_trades'],
            'avg_return': result['avg_return'],
            'median_return': result.get('median_return', 0),
            'win_rate': result['win_rate'],
            'max_gain': result.get('max_gain', 0),
            'max_loss': result.get('max_loss', 0),
            'total_return': result['total_return'],
            'full_period_return': result.get('period_return', 'N/A')
        }
        
        summary_data.append(summary_item)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/backtest_summary.csv', index=False)
    print(f"Summary saved to {output_dir}/backtest_summary.csv")
    
    # Save the overall performance results
    if 'overall_results' in locals():
        overall_df = pd.DataFrame([overall_results])
        overall_df.to_csv(f'{output_dir}/overall_performance.csv', index=False)
        print(f"Overall performance saved to {output_dir}/overall_performance.csv")
        
        # Save sentiment performance breakdown
        sentiment_df = pd.DataFrame.from_dict(sentiment_performance, orient='index')
        sentiment_df.to_csv(f'{output_dir}/sentiment_performance.csv')
        print(f"Sentiment performance saved to {output_dir}/sentiment_performance.csv")
    
    # Save detailed trade data for each ticker
    for ticker, result in results.items():
        if 'trades' in result:
            result['trades'].to_csv(f'{output_dir}/trades_{ticker}.csv', index=False)
    
    print(f"Detailed trade data saved to {output_dir}/")

def main():
    # Set the current working directory to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Load the predictions
    predictions_file = '../data/finbert_predictions_20250401_125126.csv'
    df = load_predictions(predictions_file)
    
    # Display information about the DataFrame
    print("\nColumns in the DataFrame:")
    print(df.columns.tolist())
    
    # Extract tickers and group by ticker
    ticker_groups = extract_tickers(df)
    
    # Analyze sentiment statistics by ticker
    print("\nAnalyzing sentiment statistics by ticker...")
    sentiment_stats = analyze_ticker_sentiments(df, ticker_groups)
    
    # Display sentiment statistics for top tickers (by mention count)
    print("\nSentiment Statistics for Top Tickers:")
    top_tickers = sorted(sentiment_stats.keys(), 
                         key=lambda x: sentiment_stats[x]['count'], 
                         reverse=True)[:10]
    
    for ticker in top_tickers:
        stats = sentiment_stats[ticker]
        print(f"\n{ticker} (Mentions: {stats['count']}):")
        print(f"Most common sentiment: {stats['most_common_sentiment']}")
        print("Sentiment distribution:")
        for sentiment, percentage in stats['sentiment_percentages'].items():
            print(f"  {sentiment}: {percentage:.2f}%")
    
    # Backtest strategy with a 3-day hold period
    print("\nBacktesting sentiment-based strategy (3-day hold period)...")
    # Use a lower max_tickers value due to rate limiting
    backtest_results = backtest_strategy(df, ticker_groups, days_to_hold=3, max_tickers=10)
    
    # Print backtest results summary
    print("\nBacktest Results Summary:")
    if backtest_results:
        for ticker, result in backtest_results.items():
            print(f"\n{ticker} (Signal: {result.get('sentiment', 'UNKNOWN')}):")
            print(f"Total trades: {result['total_trades']}")
            print(f"Average return per trade: {result['avg_return']:.2f}%")
            print(f"Win rate: {result['win_rate']:.2f}%")
            print(f"Total return: {result['total_return']:.2f}%")
    else:
        print("No backtest results available.")
    
    # Plot the results
    print("\nGenerating plots...")
    plot_backtest_results(backtest_results)
    
    print("\nBacktest completed.")

if __name__ == "__main__":
    main() 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Modern UI/UX: Set page config for wide layout and theme
st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide", initial_sidebar_state="expanded")

# Load the data
df = pd.read_csv('stocks_series.csv', parse_dates=['Date'])

# Unique stocks
stocks = df['name'].unique()

# Compute recommendations in background with enhanced math (weighted scores and probability-like normalization)
# Improved: Added support/resistance checks, Fibonacci signals, and now Ichimoku for more reliability
# For Ichimoku: Bullish if price > cloud, Tenkan > Kijun, Chikou > past price
if 'recommendations' not in st.session_state:
    recos = {}
    for stock in stocks:
        stock_df_temp = df[df['name'] == stock].set_index('Date').sort_index()
        if len(stock_df_temp) < 100:
            continue  # Skip short histories
        
        # Compute indicators
        def compute_rsi(series, period=14):
            delta = series.diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = compute_rsi(stock_df_temp['Close'])
        if pd.isna(rsi.iloc[-1]):
            continue
        
        def compute_macd(series, fast=12, slow=26, signal=9):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_line, signal_line, _ = compute_macd(stock_df_temp['Close'])
        
        sma_50 = stock_df_temp['Close'].rolling(window=50).mean()
        ema_20 = stock_df_temp['Close'].ewm(span=20, adjust=False).mean()
        
        # Improved: Support/Resistance
        close_vals = stock_df_temp['Close'].values
        max_idx = argrelextrema(close_vals, np.greater, order=5)[0]
        min_idx = argrelextrema(close_vals, np.less, order=5)[0]
        recent_resistances = stock_df_temp['Close'].iloc[max_idx[-3:]] if len(max_idx) >= 3 else stock_df_temp['Close'].iloc[max_idx]
        recent_supports = stock_df_temp['Close'].iloc[min_idx[-3:]] if len(min_idx) >= 3 else stock_df_temp['Close'].iloc[min_idx]
        
        # Improved: Fibonacci for recent swing
        recent_df_fib = stock_df_temp.tail(100)
        swing_high = recent_df_fib['High'].max()
        swing_low = recent_df_fib['Low'].min()
        fib_diff = swing_high - swing_low
        fib_618 = swing_low + 0.618 * fib_diff  # Golden ratio retracement
        
        # New: Ichimoku Cloud
        def compute_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
            high9 = df['High'].rolling(window=tenkan).max()
            low9 = df['Low'].rolling(window=tenkan).min()
            tenkan_sen = (high9 + low9) / 2
            
            high26 = df['High'].rolling(window=kijun).max()
            low26 = df['Low'].rolling(window=kijun).min()
            kijun_sen = (high26 + low26) / 2
            
            senkou_a = (tenkan_sen + kijun_sen) / 2
            senkou_a = senkou_a.shift(kijun)
            
            high52 = df['High'].rolling(window=senkou_b).max()
            low52 = df['Low'].rolling(window=senkou_b).min()
            senkou_b = (high52 + low52) / 2
            senkou_b = senkou_b.shift(kijun)
            
            chikou_span = df['Close'].shift(-kijun)
            
            return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span
        
        tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span = compute_ichimoku(stock_df_temp)
        
        # New: Bollinger Bands
        def compute_bollinger_bands(series, period=20, std_dev=2):
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        
        upper_bb, middle_bb, lower_bb = compute_bollinger_bands(stock_df_temp['Close'])
        
        # Enhanced scoring: Increased max_score for new signals
        score = 0
        reasons = []
        max_score = 13  # Increased for Bollinger Bands
        
        current_price = stock_df_temp['Close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            score += 3
            reasons.append("Strongly oversold (RSI < 30) - High buy probability")
        elif current_rsi < 50:
            score += 1
            reasons.append("Moderately oversold (RSI < 50) - Moderate buy signal")
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            score += 2
            reasons.append("Bullish MACD crossover - Momentum building")
        
        if current_price > sma_50.iloc[-1]:
            score += 1
            reasons.append("Price above 50-day SMA - Uptrend confirmation")
        
        if ema_20.iloc[-1] > sma_50.iloc[-1]:
            score += 1
            reasons.append("EMA20 above SMA50 - Short-term strength")
        
        # Support/Resistance signals
        if not recent_resistances.empty and current_price > recent_resistances.max():
            score += 1
            reasons.append("Price broken above recent resistance - Bullish breakout")
        if not recent_supports.empty and abs(current_price - recent_supports.max()) / current_price < 0.02:  # Within 2% of support
            score += 1
            reasons.append("Price near recent support - Potential bounce")
        
        # Fibonacci signal
        if current_price >= fib_618 and macd_line.iloc[-1] > 0:  # In uptrend context
            score += 1
            reasons.append("Price at/above 61.8% Fibonacci retracement - Bullish continuation")
        
        # Ichimoku signals
        cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
        cloud_bottom = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
        if current_price > cloud_top:
            score += 1
            reasons.append("Price above Ichimoku Cloud - Strong bullish trend")
        if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]:
            score += 1
            reasons.append("Tenkan-sen above Kijun-sen - Bullish momentum")
        
        # New: Bollinger Bands signal
        if current_price <= lower_bb.iloc[-1] and rsi.iloc[-1] < 50:
            score += 1
            reasons.append("Price near lower Bollinger Band with low RSI - Oversold bounce potential")
        
        # Math enhancement: Normalize to percentage
        prob = (score / max_score) * 100
        
        # Updated realistic targets: Use ATR for volatility-based targets
        # ATR calculation (14-period)
        high_low = stock_df_temp['High'] - stock_df_temp['Low']
        high_close = np.abs(stock_df_temp['High'] - stock_df_temp['Close'].shift())
        low_close = np.abs(stock_df_temp['Low'] - stock_df_temp['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        targets = {
            'Target 1 (1.5x ATR)': current_price + 1.5 * atr,
            'Target 2 (3x ATR)': current_price + 3 * atr
        }
        
        # Updated days estimate: Use linear regression on last 50 days for slope (daily change)
        recent_df = stock_df_temp.tail(50).copy()
        if len(recent_df) < 50:
            continue
        recent_df['Day'] = np.arange(len(recent_df))
        model = LinearRegression()
        model.fit(recent_df[['Day']], recent_df['Close'])
        slope = model.coef_[0]  # Average daily price change (drift)
        
        days_estimates = {}
        for tgt_name, tgt_price in targets.items():
            if slope > 0 and tgt_price > current_price:
                days = (tgt_price - current_price) / slope
                days_estimates[tgt_name] = max(1, round(days))  # At least 1 day
            else:
                days_estimates[tgt_name] = 'N/A (No upward trend)'
        
        # New: Swing Trading Strategies (math-based)
        # Entry: RSI < 40 near support, MACD bullish, price above cloud
        # Exit: RSI > 70 or at resistance or MACD bearish
        # Risk: Stop loss at recent support - ATR
        # Reward: Target at resistance + ATR
        swing_strategies = []
        if current_rsi < 40 and abs(current_price - recent_supports.max()) / current_price < 0.02 and macd_line.iloc[-1] > signal_line.iloc[-1] and current_price > cloud_top:
            swing_strategies.append("Buy Entry: Oversold RSI near support with bullish MACD and above Ichimoku cloud. Probability: High (based on confluence).")
        if current_rsi > 70 or current_price > recent_resistances.max() or macd_line.iloc[-1] < signal_line.iloc[-1]:
            swing_strategies.append("Sell Exit: Overbought RSI or at resistance or bearish MACD. Lock in gains.")
        
        stop_loss = current_price - atr if not recent_supports.empty else current_price - 0.05 * current_price
        expected_reward = atr * 2  # 2:1 risk-reward ratio
        swing_strategies.append(f"Risk Management: Stop loss at {stop_loss:.2f} (1x ATR below). Target reward: +{expected_reward:.2f} (2:1 ratio).")
        
        recos[stock] = {
            'score': score, 
            'prob': prob, 
            'reasons': reasons,
            'targets': targets,
            'days_estimates': days_estimates,
            'current_price': current_price,
            'stock_df': stock_df_temp,  # Store df for plotting
            'ichimoku': (tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span),  # Store for tab
            'bollinger': (upper_bb, middle_bb, lower_bb),  # Store for BB tab
            'swing_strategies': swing_strategies  # New
        }
    
    # Get top 5
    top5 = sorted(recos.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
    st.session_state.recommendations = top5
    st.session_state.all_recos = recos  # Store all for full analysis

# Sidebar for navigation and filters
with st.sidebar:
    st.title("Stock Analyzer")
    st.markdown("### Navigation")
    selected_tab = st.radio("Go to", [
        'Recommendations',
        'All Stocks Analysis',
        'Discounted Cash Flow (DCF)',
        'Earnings-Based Valuation',
        'Dividend Discount Model',
        'Financial Statement Analysis',
        'Economic Indicators',
        'Candlestick Patterns',
        'Support & Resistance',
        'Fibonacci Retracement',
        'Moving Averages',
        'MACD',
        'RSI',
        'Ichimoku Cloud',
        'Bollinger Bands',
        'Machine Learning Models'
    ])
    st.markdown("### Select Stock for Details")
    selected_stock = st.selectbox('Stock', stocks)

# Filter data for selected stock
stock_df = df[df['name'] == selected_stock].set_index('Date').sort_index()

# Main content with columns for modern layout
col1, col2 = st.columns([3, 1])

with col2:
    st.metric("Current Price", f"{stock_df['Close'].iloc[-1]:.2f}")
    st.metric("52-Week High", f"{stock_df['High'].max():.2f}")
    st.metric("52-Week Low", f"{stock_df['Low'].min():.2f}")

with col1:
    if selected_tab == 'Recommendations':
        st.header("Top 5 Stocks to Buy")
        
        # Simulation inputs
        st.subheader("Investment Simulation")
        investment_amount = st.number_input("Investment Amount ($)", min_value=0.0, value=1000.0)
        broker_fee_pct = st.number_input("Broker Fee (%)", min_value=0.0, max_value=10.0, value=0.5)
        tax_pct = st.number_input("Tax on Gains (%)", min_value=0.0, max_value=50.0, value=15.0)
        
        for stock, data in st.session_state.recommendations:
            with st.expander(f"{stock} (Score: {data['score']} | Buy Prob: {data['prob']:.0f}%)"):
                for reason in data['reasons']:
                    st.write(f"- {reason}")
                
                st.write("**Possible Targets (Volatility-Based):**")
                for tgt_name, tgt_price in data['targets'].items():
                    st.write(f"- {tgt_name}: {tgt_price:.2f}")
                    est_days = data['days_estimates'][tgt_name]
                    st.write(f"  Estimated Days to Reach (Trend-Based): {est_days}")
                    
                    # Simulate net profit
                    if isinstance(est_days, int):  # Only if estimable
                        gross_return = (tgt_price / data['current_price'] - 1) * investment_amount
                        broker_fee = investment_amount * (broker_fee_pct / 100)
                        tax = max(0, gross_return - broker_fee) * (tax_pct / 100)
                        net_profit = gross_return - broker_fee - tax
                        st.write(f"  Simulated Net Profit: {net_profit:.2f} (after {broker_fee:.2f} fee and {tax:.2f} tax)")
                
                # New: Swing Trading Strategies
                st.write("**Swing Trading Strategies (Math-Based):**")
                for strategy in data['swing_strategies']:
                    st.write(f"- {strategy}")
                
                # Modern design: Plot projected targets
                recent_df = data['stock_df'].tail(50)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], mode='lines', name='Historical Close'))
                for tgt_price in data['targets'].values():
                    fig.add_hline(y=tgt_price, line_dash="dash", line_color="green", annotation_text=f"Target {tgt_price:.2f}")
                fig.update_layout(title=f'Price Projection for {stock}', xaxis_title='Date', yaxis_title='Price', height=300)
                st.plotly_chart(fig, use_container_width=True)

    elif selected_tab == 'All Stocks Analysis':
        st.header("Full Analysis of All Stocks")
        st.write("**Meaning:** This tab provides a comprehensive overview of all stocks' scores, probabilities, and reasons based on multiple technical indicators.")
        all_df = pd.DataFrame.from_dict(st.session_state.all_recos, orient='index')
        all_df = all_df.sort_values('score', ascending=False)
        st.dataframe(all_df[['score', 'prob', 'reasons']], use_container_width=True)
        fig = px.bar(all_df.reset_index(), x='index', y='prob', title='Buy Probability by Stock')
        st.plotly_chart(fig)

    elif selected_tab == 'Discounted Cash Flow (DCF)':
        st.header("Discounted Cash Flow (DCF)")
        st.write("**Meaning:** DCF estimates a stock's intrinsic value by discounting future cash flows to present value. High value suggests undervalued stock.")
        st.write("This method requires future cash flow projections, discount rates, and terminal value, which are not available in the provided price data CSV. Additional fundamental data is needed for accurate DCF analysis.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Earnings-Based Valuation':
        st.header("Earnings-Based Valuation")
        st.write("**Meaning:** Uses ratios like P/E to compare earnings to price. Low P/E may indicate undervaluation.")
        st.write("This method uses metrics like P/E, PEG, EV/EBITDA, EPS, and growth prospects, which require earnings and financial statement data not present in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Dividend Discount Model':
        st.header("Dividend Discount Model")
        st.write("**Meaning:** Values stock based on expected dividends. Useful for dividend-paying stocks; higher yield suggests better value.")
        st.write("This method values stocks based on expected future dividends, yield, payout ratio, and growth rate. Dividend data is not in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Financial Statement Analysis':
        st.header("Financial Statement Analysis")
        st.write("**Meaning:** Analyzes balance sheets/income statements for health. High ROE/ROA indicates efficiency.")
        st.write("This involves examining income statements, balance sheets, cash flow, ROE, ROA, debt ratios – none of which are in the price data CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Economic Indicators':
        st.header("Economic Indicators")
        st.write("**Meaning:** Macro factors like GDP/inflation affect markets. Rising GDP may boost stocks.")
        st.write("Incorporates macroeconomic factors like GDP, inflation, interest rates, PMI. These external indicators are not in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Candlestick Patterns':
        st.header("Candlestick Patterns")
        st.write("**Meaning:** Visual representation of price action. Bullish patterns (e.g., hammer) suggest uptrends; bearish (e.g., shooting star) downtrends.")
        fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                            open=stock_df['Open'],
                                            high=stock_df['High'],
                                            low=stock_df['Low'],
                                            close=stock_df['Close'])])
        fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Result: Inspect the chart for patterns like dojis (Open ≈ Close) or hammers (small body, long lower wick).")

    elif selected_tab == 'Support & Resistance':
        st.header("Support & Resistance")
        st.write("**Meaning:** Support is a floor price (buyers enter); resistance
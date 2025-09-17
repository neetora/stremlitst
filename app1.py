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
# Updated: More realistic targets using volatility (ATR) and days using linear regression slope
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
        
        # Enhanced scoring: Weighted and normalized (e.g., RSI weight 0.4, MACD 0.3, MAs 0.3)
        # Add math: Normalize scores to 0-100% 'probability' of buy signal based on thresholds
        score = 0
        reasons = []
        max_score = 7  # For normalization
        
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            score += 3  # High weight for strong oversold
            reasons.append("Strongly oversold (RSI < 30) - High buy probability")
        elif current_rsi < 50:
            score += 1
            reasons.append("Moderately oversold (RSI < 50) - Moderate buy signal")
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            score += 2
            reasons.append("Bullish MACD crossover - Momentum building")
        
        if stock_df_temp['Close'].iloc[-1] > sma_50.iloc[-1]:
            score += 1
            reasons.append("Price above 50-day SMA - Uptrend confirmation")
        
        if ema_20.iloc[-1] > sma_50.iloc[-1]:
            score += 1
            reasons.append("EMA20 above SMA50 - Short-term strength")
        
        # Math enhancement: Normalize to percentage
        prob = (score / max_score) * 100
        
        # Updated realistic targets: Use ATR for volatility-based targets
        # ATR calculation (14-period)
        high_low = stock_df_temp['High'] - stock_df_temp['Low']
        high_close = np.abs(stock_df_temp['High'] - stock_df_temp['Close'].shift())
        low_close = np.abs(stock_df_temp['Low'] - stock_df_temp['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        current_price = stock_df_temp['Close'].iloc[-1]
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
        
        recos[stock] = {
            'score': score, 
            'prob': prob, 
            'reasons': reasons,
            'targets': targets,
            'days_estimates': days_estimates,
            'current_price': current_price,
            'stock_df': stock_df_temp  # Store df for plotting
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
        all_df = pd.DataFrame.from_dict(st.session_state.all_recos, orient='index')
        all_df = all_df.sort_values('score', ascending=False)
        st.dataframe(all_df[['score', 'prob', 'reasons']], use_container_width=True)
        fig = px.bar(all_df.reset_index(), x='index', y='prob', title='Buy Probability by Stock')
        st.plotly_chart(fig)

    elif selected_tab == 'Discounted Cash Flow (DCF)':
        st.header("Discounted Cash Flow (DCF)")
        st.write("This method requires future cash flow projections, discount rates, and terminal value, which are not available in the provided price data CSV. Additional fundamental data is needed for accurate DCF analysis.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Earnings-Based Valuation':
        st.header("Earnings-Based Valuation")
        st.write("This method uses metrics like P/E, PEG, EV/EBITDA, EPS, and growth prospects, which require earnings and financial statement data not present in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Dividend Discount Model':
        st.header("Dividend Discount Model")
        st.write("This method values stocks based on expected future dividends, yield, payout ratio, and growth rate. Dividend data is not in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Financial Statement Analysis':
        st.header("Financial Statement Analysis")
        st.write("This involves examining income statements, balance sheets, cash flow, ROE, ROA, debt ratios – none of which are in the price data CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Economic Indicators':
        st.header("Economic Indicators")
        st.write("Incorporates macroeconomic factors like GDP, inflation, interest rates, PMI. These external indicators are not in the CSV.")
        st.write("Result: Not applicable with current data.")

    elif selected_tab == 'Candlestick Patterns':
        st.header("Candlestick Patterns")
        st.write("Visual patterns indicating price movement and trend changes (e.g., Doji, Hammer, Shooting Star).")
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
        st.write("Price levels where stocks historically stall or reverse.")
        close = stock_df['Close'].values
        max_idx = argrelextrema(close, np.greater, order=5)[0]
        min_idx = argrelextrema(close, np.less, order=5)[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=stock_df.index[max_idx], y=stock_df['Close'].iloc[max_idx], mode='markers', marker=dict(color='red'), name='Resistance'))
        fig.add_trace(go.Scatter(x=stock_df.index[min_idx], y=stock_df['Close'].iloc[min_idx], mode='markers', marker=dict(color='green'), name='Support'))
        fig.update_layout(title='Support and Resistance', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Key Supports:", stock_df['Close'].iloc[min_idx].tolist())
        st.write("Key Resistances:", stock_df['Close'].iloc[max_idx].tolist())
        
        # Add Possible Targets and Prediction Days
        if selected_stock in st.session_state.all_recos:
            data = st.session_state.all_recos[selected_stock]
            st.subheader("Possible Targets (Volatility-Based)")
            for tgt_name, tgt_price in data['targets'].items():
                st.write(f"- {tgt_name}: {tgt_price:.2f}")
                est_days = data['days_estimates'][tgt_name]
                st.write(f"  Estimated Days to Reach (Trend-Based): {est_days}")

    elif selected_tab == 'Fibonacci Retracement':
        st.header("Fibonacci Retracement")
        st.write("Identifies potential reversal points using mathematical ratios.")
        recent_df = stock_df.tail(100)
        high = recent_df['High'].max()
        low = recent_df['Low'].min()
        diff = high - low
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_levels = [high - level * diff for level in levels]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close Price'))
        for lvl, fib in zip(levels, fib_levels):
            fig.add_hline(y=fib, line_dash="dash", line_color="orange", annotation_text=f"{lvl*100:.1f}%")
        fig.update_layout(title='Fibonacci Levels', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Fibonacci Levels:", {f'{levels[i]*100}%': fib_levels[i] for i in range(len(levels))})

    elif selected_tab == 'Moving Averages':
        st.header("Moving Averages")
        st.write("Smooths price data to identify trends.")
        sma_20 = stock_df['Close'].rolling(window=20).mean()
        ema_20 = stock_df['Close'].ewm(span=20, adjust=False).mean()
        sma_50 = stock_df['Close'].rolling(window=50).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=sma_20, mode='lines', name='SMA 20'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=ema_20, mode='lines', name='EMA 20'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=sma_50, mode='lines', name='SMA 50'))
        fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Current SMA 20:", sma_20.iloc[-1])
        st.write("Current EMA 20:", ema_20.iloc[-1])
        st.write("Current SMA 50:", sma_50.iloc[-1])

    elif selected_tab == 'MACD':
        st.header("MACD")
        st.write("Tracks trend momentum.")
        def compute_macd(series, fast=12, slow=26, signal=9):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_line, signal_line, histogram = compute_macd(stock_df['Close'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=macd_line, mode='lines', name='MACD Line'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=signal_line, mode='lines', name='Signal Line'))
        fig.add_trace(go.Bar(x=stock_df.index, y=histogram, name='Histogram', marker_color='gray'))
        fig.update_layout(title='MACD', xaxis_title='Date')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Current MACD:", macd_line.iloc[-1])
        st.write("Current Signal:", signal_line.iloc[-1])

    elif selected_tab == 'RSI':
        st.header("RSI")
        st.write("Identifies overbought/oversold conditions.")
        def compute_rsi(series, period=14):
            delta = series.diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = compute_rsi(stock_df['Close'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=rsi, mode='lines', name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.update_layout(title='RSI', xaxis_title='Date', yaxis_title='RSI Value')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Current RSI:", rsi.iloc[-1])
        if rsi.iloc[-1] > 70:
            st.write("Status: Overbought")
        elif rsi.iloc[-1] < 30:
            st.write("Status: Oversold")
        else:
            st.write("Status: Neutral")

    elif selected_tab == 'Machine Learning Models':
        st.header("Machine Learning Models")
        st.write("Simple LSTM for price prediction.")
        if len(stock_df) < 60:
            st.write("Insufficient data for ML training.")
        else:
            prices = stock_df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            
            seq_length = 60
            X, y = [], []
            for i in range(len(scaled_prices) - seq_length):
                X.append(scaled_prices[i:i+seq_length])
                y.append(scaled_prices[i+seq_length])
            X, y = np.array(X), np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            class LSTMModel(nn.Module):
                def __init__(self):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(1, 50, num_layers=1, batch_first=True)
                    self.fc = nn.Linear(50, 1)
                
                def forward(self, x):
                    h0 = torch.zeros(1, x.size(0), 50)
                    c0 = torch.zeros(1, x.size(0), 50)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            epochs = 10
            for epoch in range(epochs):
                inputs = torch.from_numpy(X_train).float()
                targets = torch.from_numpy(y_train).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                test_inputs = torch.from_numpy(X_test).float()
                predicted = model(test_inputs).numpy()
                predicted = scaler.inverse_transform(predicted)
                actual = scaler.inverse_transform(y_test)
            
            test_df = pd.DataFrame({'Actual': actual.flatten(), 'Predicted': predicted.flatten()})
            fig = px.line(test_df, title='LSTM Price Prediction', labels={'index': 'Time Steps', 'value': 'Price'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Mean Squared Error:", np.mean((predicted - actual)**2))
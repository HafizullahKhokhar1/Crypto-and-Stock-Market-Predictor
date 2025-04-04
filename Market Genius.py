import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import time
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(page_title='Market Genius', layout='wide', page_icon='📈')
st.sidebar.title("🎛️ Control Panel")

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .metric-card {border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .prediction-card {border-left: 4px solid #4e73df; padding: 1rem; 
                     margin-bottom: 1rem; background-color: #f8f9fa;}
    .news-card {border-left: 4px solid #6c757d; padding: 1rem; margin: 0.5rem 0;}
    .positive {color: #28a745;}
    .negative {color: #dc3545;}
</style>
""", unsafe_allow_html=True)

# Initialize News API
try:
    newsapi_key = os.getenv('NEWS_API_KEY') or 'Add Your NewsAPI'  # Fallback to direct key
    if newsapi_key:
        newsapi = NewsApiClient(api_key=newsapi_key)
    else:
        newsapi = None
        st.sidebar.warning("News API key not found")
except Exception as e:
    st.sidebar.warning(f"News API configuration error: {str(e)}")
    newsapi = None

# Load Model
try:
    model = load_model("keras_model.h5")
except Exception:
    model = None
    st.sidebar.error("❌ Model not found. Upload a valid keras_model.h5")

# Enhanced Prediction Functions
def predict_future_prices(data, prediction_days, model):
    if model is None or data.empty:
        return None, None, None
    
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        x = []
        sequence_length = 60
        for i in range(sequence_length, len(scaled_data)):
            x.append(scaled_data[i-sequence_length:i, 0])
        x = np.array(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        last_sequence = x[-1]
        current_batch = last_sequence.reshape(1, sequence_length, 1)
        
        predictions = []
        confidence_scores = []
        for _ in range(prediction_days):
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            confidence_scores.append(np.max(current_pred))
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [data.index[-1] + dt.timedelta(days=i) for i in range(1, prediction_days+1)]
        
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions.flatten(),
            'Confidence': np.array(confidence_scores).flatten()
        }).set_index('Date')
        
        return prediction_df, predictions[-1][0], np.mean(confidence_scores)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_trading_suggestion(current_price, predicted_prices, confidence):
    if predicted_prices is None or current_price is None or not predicted_prices.any():
        return "Hold", "Insufficient data", "#6c757d", 0
    
    try:
        prediction_days = len(predicted_prices)
        short_term_days = max(1, prediction_days // 3)
        
        short_term_change = (predicted_prices[short_term_days-1] - current_price) / current_price * 100
        long_term_change = (predicted_prices[-1] - current_price) / current_price * 100
        
        confidence_score = min(100, max(0, int(confidence * 100)))
        
        if long_term_change > 7 and confidence_score > 70:
            return "🔥 Strong Buy", f"{short_term_days}-{prediction_days} days", "#28a745", confidence_score
        elif long_term_change > 5:
            return "📈 Buy", f"{prediction_days} days", "#7ac29a", confidence_score
        elif long_term_change < -7 and confidence_score > 70:
            return "💣 Strong Sell", f"Immediate", "#dc3545", confidence_score
        elif long_term_change < -5:
            return "📉 Sell", f"{prediction_days} days", "#f8b7cd", confidence_score
        elif abs(short_term_change) > 3 and confidence_score > 65:
            if short_term_change > 0:
                return "⚡ Short Buy", f"{short_term_days} days", "#7ac29a", confidence_score
            else:
                return "⚡ Short Sell", f"{short_term_days} days", "#f8b7cd", confidence_score
        else:
            return "🤝 Hold", "Wait", "#6c757d", confidence_score
    except Exception:
        return "Hold", "Error", "#6c757d", 0

# Market Fear & Greed Index
def get_market_sentiment():
    return {
        'score': np.random.randint(20, 80),
        'sentiment': ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'][np.random.randint(0, 5)],
        'color': ['#dc3545', '#ffc107', '#6c757d', '#28a745', '#218838'][np.random.randint(0, 5)]
    }

# Sidebar Controls
page = st.sidebar.radio("Select Market", ["📊 Stock Market", "₿ Cryptocurrency"])

if page == "📊 Stock Market":
    st.title("📊 Stock Market Genius")
    
    with st.sidebar.expander("⚙️ Stock Parameters", expanded=True):
        stock_options = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", 
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE"
        ]
        STOCK = st.selectbox("Select Stock Ticker", stock_options, index=0)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", dt.date(2022, 1, 1))
        with col2:
            end_date = st.date_input("End Date", dt.date.today())
        prediction_days = st.slider("Days to predict", 1, 30, 7)
        st.markdown("---")
        st.markdown("**Technical Indicators**")
        show_rsi = st.checkbox("Show RSI", True)
        show_macd = st.checkbox("Show MACD", True)
        show_sma = st.checkbox("Show SMA/EMA", True)

    # Main Content
    @st.cache_data(ttl=3600)
    def load_stock_data(ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Calculate indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp12 = df['Close'].ewm(span=12, adjust=False).mean()
                exp26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp12 - exp26
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    df = load_stock_data(STOCK, start_date, end_date)

    if not df.empty:
        try:
            live_stock = yf.Ticker(STOCK).history(period='1d')
            current_price = live_stock['Close'].iloc[-1] if not live_stock.empty else None
            previous_close = live_stock['Close'].iloc[-2] if len(live_stock) > 1 else None
            price_change = current_price - previous_close if current_price and previous_close else 0
        except Exception:
            current_price, previous_close, price_change = None, None, None

        # Market Overview Cards
        st.subheader("📌 Market Overview")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<div class='metric-card'><h3>Current Price</h3><h2>{f'${current_price:.2f}' if current_price else 'N/A'}</h2></div>", 
                       unsafe_allow_html=True)
        with cols[1]:
            change_pct = (price_change/previous_close*100) if previous_close and price_change else 0
            change_class = "positive" if change_pct >= 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Daily Change</h3>
                <h2 class='{change_class}'>{f'${price_change:+.2f}' if price_change else 'N/A'}</h2>
                <p class='{change_class}'>{f'{change_pct:+.2f}%' if previous_close and price_change else ''}</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            min_price = float(df['Close'].min())
            max_price = float(df['Close'].max())
            st.markdown(f"""
            <div class='metric-card'>
                <h3>52 Week Range</h3>
                <h4>${min_price:.2f} - ${max_price:.2f}</h4>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            sentiment = get_market_sentiment()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Market Sentiment</h3>
                <h4>{sentiment['sentiment']}</h4>
                <p>Score: {sentiment['score']}/100</p>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Price Chart
        st.subheader("📈 Interactive Price Analysis")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price'
        ))
        
        if show_sma:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='orange')))
        
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators
        if show_rsi or show_macd:
            st.subheader("📊 Technical Indicators")
            
        if show_rsi:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.update_layout(height=250, showlegend=False, title="RSI (14 days)")
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        if show_macd:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')))
            macd_fig.update_layout(height=250, showlegend=False, title="MACD")
            st.plotly_chart(macd_fig, use_container_width=True)

        # Prediction Section
        if model is not None:
            st.subheader("🔮 AI Price Prediction")
            with st.spinner('Generating predictions...'):
                prediction_df, final_pred, confidence = predict_future_prices(df, prediction_days, model)
                
                if prediction_df is not None:
                    fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Price',
                                title=f"{prediction_days}-Day Price Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    suggestion, timeframe, color, conf_score = generate_trading_suggestion(
                        current_price, prediction_df['Predicted Price'].values, confidence)
                    
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem;">
                            <h2 style="color:white; text-align:center;">{suggestion}</h2>
                            <p style="color:white; text-align:center;">
                                Timeframe: {timeframe} | Confidence: {conf_score}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        prediction_df.style.format({
                            "Predicted Price": "${:.2f}",
                            "Confidence": "{:.1%}"
                        }).background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                        height=min(300, len(prediction_df)*35)
                    )

                    # News Section - Now properly integrated with error handling
                    if newsapi:
                        try:
                            st.subheader("📰 Latest Market News")
                            news = newsapi.get_everything(q=STOCK, language='en', sort_by='publishedAt', page_size=3)
                            
                            if news and news.get('articles'):
                                for article in news['articles']:
                                    if article['title'] and article['description']:
                                        st.markdown(f"""
                                        <div class='news-card'>
                                            <h4>{article['title']}</h4>
                                            <p><small>{article['source']['name']} • {article['publishedAt'][:10]}</small></p>
                                            <p>{article['description']}</p>
                                            <a href="{article['url']}" target="_blank">Read more →</a>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No recent news found for this stock")
                        except Exception as e:
                            st.warning(f"Could not load news: {str(e)}")
                    else:
                        st.warning("News API not configured - please check your API key")

elif page == "₿ Cryptocurrency":
    st.title("₿ Cryptocurrency Genius")
    
    with st.sidebar.expander("⚙️ Crypto Parameters", expanded=True):
        crypto_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"]
        crypto_symbol = st.selectbox("Select Crypto Pair", crypto_options, index=0)
        prediction_days = st.slider("Days to predict", 1, 30, 7)
        st.markdown("---")
        st.markdown("**Crypto Metrics**")
        show_volume = st.checkbox("Show Trading Volume", True)
        show_volatility = st.checkbox("Show Volatility", True)
    
    # Main Content
    @st.cache_data(ttl=600)
    def fetch_crypto_data(symbol, limit=365):
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
            data = requests.get(url).json()
            df = pd.DataFrame(data, columns=[
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'CloseTime', 'QuoteAssetVolume', 'Trades', 
                'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignore'
            ])
            df['Time'] = pd.to_datetime(df['Time'], unit='ms')
            df.set_index('Time', inplace=True)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Calculate volatility
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=7).std() * np.sqrt(365)
            
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    crypto_data = fetch_crypto_data(crypto_symbol)
    
    if not crypto_data.empty:
        current_price = crypto_data['Close'].iloc[-1]
        prev_price = crypto_data['Close'].iloc[-2]
        price_change = current_price - prev_price
        change_pct = (price_change/prev_price)*100
        
        # Crypto Overview Cards
        st.subheader("📌 Crypto Overview")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<div class='metric-card'><h3>Current Price</h3><h2>${current_price:,.2f}</h2></div>", 
                       unsafe_allow_html=True)
        with cols[1]:
            change_class = "positive" if price_change >= 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>24h Change</h3>
                <h2 class='{change_class}'>${price_change:+,.2f}</h2>
                <p class='{change_class}'>{change_pct:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            vol_24h = crypto_data['Volume'].iloc[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h3>24h Volume</h3>
                <h4>${vol_24h:,.0f}</h4>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            volatility = crypto_data['Volatility'].iloc[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Volatility</h3>
                <h4>{volatility:.2%}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Crypto Chart
        st.subheader("📈 Crypto Price Analysis")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=crypto_data.index,
            open=crypto_data['Open'],
            high=crypto_data['High'],
            low=crypto_data['Low'],
            close=crypto_data['Close'],
            name='Price'
        ))
        
        if show_volume:
            fig.add_trace(go.Bar(
                x=crypto_data.index,
                y=crypto_data['Volume'],
                name='Volume',
                marker_color='rgba(100, 149, 237, 0.6)',
                yaxis='y2'
            ))
        
        if show_volatility:
            fig.add_trace(go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Volatility'],
                name='Volatility',
                line=dict(color='red', width=1),
                yaxis='y3'
            ))
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            yaxis=dict(title='Price', domain=[0.3, 1.0]),
            yaxis2=dict(title='Volume', domain=[0.15, 0.3], showgrid=False),
            yaxis3=dict(title='Volatility', domain=[0.0, 0.15], showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Section
        if model is not None:
            st.subheader("🔮 AI Price Prediction")
            with st.spinner('Generating predictions...'):
                prediction_df, final_pred, confidence = predict_future_prices(
                    crypto_data, prediction_days, model)
                
                if prediction_df is not None:
                    fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Price',
                                title=f"{prediction_days}-Day Price Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    suggestion, timeframe, color, conf_score = generate_trading_suggestion(
                        current_price, prediction_df['Predicted Price'].values, confidence)
                    
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem;">
                            <h2 style="color:white; text-align:center;">{suggestion}</h2>
                            <p style="color:white; text-align:center;">
                                Timeframe: {timeframe} | Confidence: {conf_score}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        prediction_df.style.format({
                            "Predicted Price": "${:.2f}",
                            "Confidence": "{:.1%}"
                        }).background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                        height=min(300, len(prediction_df)*35))
        
        # Crypto Fear & Greed Index
        st.subheader("😨😊 Crypto Fear & Greed Index")
        col1, col2 = st.columns([1, 3])
        with col1:
            sentiment = get_market_sentiment()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Sentiment"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': sentiment['color']},
                    'steps': [
                        {'range': [0, 25], 'color': "#dc3545"},
                        {'range': [25, 50], 'color': "#ffc107"},
                        {'range': [50, 75], 'color': "#28a745"},
                        {'range': [75, 100], 'color': "#218838"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### What is the Crypto Fear & Greed Index?
            - **0-24**: Extreme Fear (Potential buying opportunity)
            - **25-49**: Fear (Market may be undervalued)
            - **50-74**: Greed (Market may be overvalued)
            - **75-100**: Extreme Greed (Potential market top)
            
            This index measures emotions and sentiments from different sources including:
            - Volatility (25%)
            - Market Momentum/Volume (25%)
            - Social Media (15%)
            - Surveys (10%)
            - Dominance (10%)
            - Trends (15%)
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
ℹ️ **About Market Genius**  
Advanced AI-powered market analysis tool providing:  
- Real-time price predictions  
- Technical indicators  
- Trading recommendations  
- Market sentiment analysis
""")

# Add watermark
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>Made with ❤️ using Streamlit | [Report Issues](https://github.com/HafizullahKhokhar1/Crypto-and-Stock-Market-Predictor/issues)</small>
""", unsafe_allow_html=True)

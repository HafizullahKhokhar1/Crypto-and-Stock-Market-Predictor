# 📈 Market Genius - AI-Powered Stock & Crypto Market Predictor

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43.2-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Advanced Machine Learning Platform for Real-Time Market Analysis and Price Predictions**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Technology](#-technology-stack) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

**Market Genius** is a cutting-edge, AI-powered financial analysis platform that leverages deep learning to provide real-time stock and cryptocurrency market predictions. Built with state-of-the-art LSTM (Long Short-Term Memory) neural networks, this tool empowers traders and investors with actionable insights, technical indicators, and intelligent trading recommendations.

### 🌟 Why Market Genius?

- **🤖 AI-Driven Predictions**: Advanced LSTM neural network model trained on historical market data
- **📊 Real-Time Analysis**: Live price tracking with interactive visualizations
- **🎯 Smart Recommendations**: Automated buy/sell/hold suggestions with confidence scores
- **📰 Market Intelligence**: Integrated news feed for informed decision-making
- **🔍 Technical Analysis**: Comprehensive indicators including RSI, MACD, SMA, and EMA
- **💹 Multi-Asset Support**: Trade both stocks and cryptocurrencies from one platform

---

## ✨ Features

### 📊 **Stock Market Analysis**
- **16+ Popular Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, and more
- **Interactive Candlestick Charts**: Professional-grade price visualization
- **Technical Indicators**:
  - 📈 RSI (Relative Strength Index) - Momentum indicator
  - 📉 MACD (Moving Average Convergence Divergence) - Trend indicator
  - 🔵 SMA/EMA (Simple & Exponential Moving Averages) - Trend analysis
- **52-Week Range Tracking**: Historical price boundaries
- **Market Sentiment Analysis**: Fear & Greed Index integration
- **Real-Time News Integration**: Latest market-moving headlines via NewsAPI

### ₿ **Cryptocurrency Trading**
- **Top Crypto Pairs**: BTC/USDT, ETH/USDT, BNB/USDT, XRP/USDT, SOL/USDT, DOGE/USDT
- **Binance API Integration**: Real-time cryptocurrency data
- **24-Hour Metrics**: Price changes, volume, and volatility tracking
- **Advanced Volatility Analysis**: Risk assessment for crypto assets
- **Fear & Greed Gauge**: Crypto market sentiment visualization

### 🔮 **AI Price Prediction Engine**
- **Custom Prediction Timeframes**: 1-30 day forecasts
- **LSTM Neural Network**: Deep learning model for time series forecasting
- **Confidence Scoring**: AI certainty metrics for each prediction
- **Automated Trading Signals**:
  - 🔥 Strong Buy/Sell alerts
  - 📈 Buy/Sell recommendations
  - ⚡ Short-term trading opportunities
  - 🤝 Hold suggestions

### 🎨 **User Experience**
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Interactive Plotly Charts**: Zoom, pan, and explore data dynamically
- **Dark/Light Mode Ready**: Professional color schemes
- **Real-Time Updates**: Live data refresh capabilities
- **Customizable Parameters**: Tailor analysis to your needs

---

## 🛠️ Technology Stack

### **Core Technologies**
| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Primary programming language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning framework for LSTM model |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interactive web application framework |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation and analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |

### **Key Libraries**
- **yfinance** (v0.2.54) - Yahoo Finance API for stock data
- **plotly** (v6.0.1) - Interactive data visualizations
- **scikit-learn** (v1.6.1) - Machine learning preprocessing
- **newsapi-python** (v0.2.7) - Financial news aggregation
- **requests** (v2.32.3) - HTTP requests for crypto data

### **Machine Learning Architecture**
- **Model Type**: LSTM (Long Short-Term Memory) Recurrent Neural Network
- **Training Data**: Historical price sequences (60-day windows)
- **Normalization**: MinMax scaling for optimal performance
- **Prediction Method**: Sequential forecasting with sliding windows

---

## 📦 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/HafizullahKhokhar1/Crypto-and-Stock-Market-Predictor.git
cd Crypto-and-Stock-Market-Predictor
```

### **Step 2: Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Configure API Keys**
Create a `.env` file in the project root directory:
```env
NEWS_API_KEY=your_newsapi_key_here
```

**Get Your NewsAPI Key**: 
1. Visit [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Copy your API key
4. Paste it in the `.env` file

---

## 🚀 Usage

### **Launch the Application**
```bash
streamlit run "Market Genius.py"
```

The application will open automatically in your default web browser at `http://localhost:8501`

### **Quick Start Guide**

#### **1️⃣ Select Market Type**
- Choose between **📊 Stock Market** or **₿ Cryptocurrency** from the sidebar

#### **2️⃣ Configure Parameters**
**For Stocks:**
- Select stock ticker (e.g., AAPL, MSFT)
- Set date range for historical data
- Choose prediction timeframe (1-30 days)
- Toggle technical indicators (RSI, MACD, SMA/EMA)

**For Crypto:**
- Select cryptocurrency pair (e.g., BTCUSDT)
- Set prediction days
- Enable volume and volatility displays

#### **3️⃣ Analyze Results**
- **Market Overview**: Current price, daily changes, volume
- **Interactive Charts**: Candlestick patterns and price movements
- **Technical Indicators**: RSI, MACD, and moving averages
- **AI Predictions**: Future price forecasts with confidence scores
- **Trading Signals**: Buy/Sell/Hold recommendations
- **Latest News**: Market-relevant headlines (stocks only)

---

## 📊 Model Information

### **LSTM Neural Network Architecture**
The prediction model (`keras_model.h5`) is built using TensorFlow and trained on historical market data:

**Training Specifications:**
- **Input Sequence Length**: 60 days of historical prices
- **Feature Normalization**: MinMax scaling (0-1 range)
- **Prediction Horizon**: 1-30 days forward
- **Confidence Metric**: Model certainty scores for each prediction

**How It Works:**
1. Loads 60-day historical price sequences
2. Normalizes data using MinMaxScaler
3. Processes sequences through LSTM layers
4. Generates future price predictions
5. Denormalizes outputs to actual price values
6. Calculates confidence scores based on model certainty

---

## 🎨 Screenshots & Demo

### Stock Market Dashboard
The stock market interface provides comprehensive analysis with real-time data, technical indicators, and AI-powered predictions.

### Cryptocurrency Analysis
Track cryptocurrency markets with advanced volatility metrics and sentiment indicators.

### AI Prediction Engine
View multi-day price forecasts with confidence scores and automated trading recommendations.

---

## 🔧 Configuration

### **Environment Variables**
| Variable | Description | Required |
|----------|-------------|----------|
| `NEWS_API_KEY` | API key from NewsAPI.org for financial news | Optional |

### **Customization Options**
- Modify stock/crypto ticker lists in `Market Genius.py`
- Adjust prediction sequence length (default: 60 days)
- Customize technical indicator parameters
- Modify color schemes and styling in CSS section

---

## 📈 API Integration

### **Data Sources**
1. **Yahoo Finance** (yfinance): Stock market data
2. **Binance API**: Cryptocurrency real-time prices
3. **NewsAPI**: Financial news and headlines

### **Rate Limits**
- NewsAPI Free Tier: 100 requests/day
- Binance API: 1200 requests/minute
- Yahoo Finance: No official limit (use responsibly)

---

## 🤝 Contributing

We welcome contributions from the community! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on:
- Reporting issues
- Submitting pull requests
- Code style guidelines
- Development workflow

### **How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TensorFlow Team** - Deep learning framework
- **Streamlit Community** - Web application framework
- **Yahoo Finance** - Stock market data provider
- **Binance** - Cryptocurrency data API
- **NewsAPI** - Financial news aggregation

---

## 📞 Contact & Support

- **Developer**: Hafizullah Khokhar
- **GitHub**: [@HafizullahKhokhar1](https://github.com/HafizullahKhokhar1)
- **Issues**: [Report bugs or request features](https://github.com/HafizullahKhokhar1/Crypto-and-Stock-Market-Predictor/issues)

---

## ⚠️ Disclaimer

**Important**: This tool is for educational and research purposes only. Market predictions are based on historical data and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.

---

## 🔑 Keywords

`stock market prediction` `cryptocurrency analysis` `machine learning trading` `LSTM neural network` `technical indicators` `AI trading bot` `financial forecasting` `real-time stock analysis` `crypto price prediction` `algorithmic trading` `Python trading platform` `deep learning finance` `market sentiment analysis` `TensorFlow finance` `automated trading signals` `yfinance` `Binance API` `RSI MACD indicators` `Streamlit dashboard` `financial data science`

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by [Hafizullah Khokhar](https://github.com/HafizullahKhokhar1)**

</div>

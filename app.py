import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load your trained model (you'll need to save it first)
@st.cache_resource
def load_model():
    try:
        # Load your best model - adjust the filename based on your saved model
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        # If no saved model, create a simple one for demonstration
        st.warning("No saved model found. Using a demo model.")
        return None, None

def create_features(data_dict):
    """Create features from input data"""
    # Convert input to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Create technical indicators
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close_Lag1'] - df['Open']
    df['Price_Change_Pct'] = (df['Close_Lag1'] - df['Open']) / df['Open'] * 100
    
    # Simple moving averages (using lag values as approximation)
    df['MA_5'] = df['Close_Lag1']  # Simplified for demo
    df['MA_10'] = df['Close_Lag1']
    df['MA_20'] = df['Close_Lag1']
    
    # Volatility (simplified)
    df['Volatility_5'] = abs(df['High'] - df['Low']) / df['Close_Lag1'] * 100
    df['Volatility_10'] = df['Volatility_5']
    
    # Momentum
    df['Momentum_5'] = df['Close_Lag1'] - df['Close_Lag2']
    df['Momentum_10'] = df['Close_Lag1'] - df['Close_Lag3']
    
    # Volume indicators
    df['Volume_MA_5'] = df['Volume']
    df['Volume_Ratio'] = 1.0  # Simplified
    
    # RSI (simplified calculation)
    price_change = df['Close_Lag1'] - df['Close_Lag2']
    df['RSI'] = 50 + (price_change / df['Close_Lag1'] * 100).clip(-50, 50)
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("### Predict tomorrow's stock closing price using machine learning")
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Stock Data")
    st.sidebar.markdown("Enter the current stock information:")
    
    # Input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        open_price = st.number_input(
            "Open Price ($)", 
            min_value=0.01, 
            value=150.00, 
            step=0.01,
            help="Opening price of the stock"
        )
        
        high_price = st.number_input(
            "High Price ($)", 
            min_value=0.01, 
            value=155.00, 
            step=0.01,
            help="Highest price during the day"
        )
        
        low_price = st.number_input(
            "Low Price ($)", 
            min_value=0.01, 
            value=148.00, 
            step=0.01,
            help="Lowest price during the day"
        )
    
    with col2:
        volume = st.number_input(
            "Volume", 
            min_value=1, 
            value=1000000, 
            step=1000,
            help="Number of shares traded"
        )
        
        adj_close = st.number_input(
            "Adjusted Close ($)", 
            min_value=0.01, 
            value=152.00, 
            step=0.01,
            help="Adjusted closing price"
        )
    
    # Historical prices (lag features)
    st.sidebar.subheader("📅 Historical Prices")
    close_lag1 = st.sidebar.number_input(
        "Yesterday's Close ($)", 
        min_value=0.01, 
        value=151.00, 
        step=0.01,
        help="Previous day closing price"
    )
    
    close_lag2 = st.sidebar.number_input(
        "2 Days Ago Close ($)", 
        min_value=0.01, 
        value=149.00, 
        step=0.01,
        help="Closing price 2 days ago"
    )
    
    close_lag3 = st.sidebar.number_input(
        "3 Days Ago Close ($)", 
        min_value=0.01, 
        value=147.00, 
        step=0.01,
        help="Closing price 3 days ago"
    )
    
    # Validation
    if high_price < max(open_price, low_price, adj_close):
        st.sidebar.error("High price should be the highest value!")
        return
    
    if low_price > min(open_price, high_price, adj_close):
        st.sidebar.error("Low price should be the lowest value!")
        return
    
    # Predict button
    predict_button = st.sidebar.button("🔮 Predict Stock Price", type="primary")
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if predict_button:
            # Prepare input data
            input_data = {
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Volume': volume,
                'Adjusted Close': adj_close,
                'Close_Lag1': close_lag1,
                'Close_Lag2': close_lag2,
                'Close_Lag3': close_lag3
            }
            
            # Create features
            features_df = create_features(input_data)
            
            # Feature columns (same as in your training)
            feature_columns = [
                'Open', 'High', 'Low', 'Volume', 'Adjusted Close',
                'Price_Range', 'Price_Change', 'Price_Change_Pct',
                'MA_5', 'MA_10', 'MA_20',
                'Volatility_5', 'Volatility_10',
                'Momentum_5', 'Momentum_10',
                'Volume_MA_5', 'Volume_Ratio',
                'Close_Lag1', 'Close_Lag2', 'Close_Lag3',
                'RSI'
            ]
            
            # Prepare features for prediction
            X_input = features_df[feature_columns]
            
            # Make prediction
            if model is not None and scaler is not None:
                try:
                    # Scale features if needed
                    X_scaled = scaler.transform(X_input)
                    prediction = model.predict(X_scaled)[0]
                    
                    # Display prediction
                    st.markdown(f'<div class="prediction-result">Predicted Closing Price: ${prediction:.2f}</div>', 
                              unsafe_allow_html=True)
                    
                    # Calculate prediction insights
                    price_change = prediction - close_lag1
                    price_change_pct = (price_change / close_lag1) * 100
                    
                    # Display metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            label="Price Change",
                            value=f"${price_change:.2f}",
                            delta=f"{price_change_pct:.2f}%"
                        )
                    
                    with col_b:
                        st.metric(
                            label="Current Price",
                            value=f"${close_lag1:.2f}"
                        )
                    
                    with col_c:
                        direction = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
                        st.metric(
                            label="Trend",
                            value=direction
                        )
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    # Fallback simple prediction
                    simple_prediction = close_lag1 * (1 + np.random.normal(0, 0.02))
                    st.markdown(f'<div class="prediction-result">Estimated Price: ${simple_prediction:.2f}</div>', 
                              unsafe_allow_html=True)
            else:
                # Demo prediction when no model is loaded
                demo_prediction = close_lag1 * (1 + (high_price - low_price) / close_lag1 * 0.1)
                st.markdown(f'<div class="prediction-result">Demo Prediction: ${demo_prediction:.2f}</div>', 
                          unsafe_allow_html=True)
                st.info("This is a demo prediction. Load your trained model for accurate results.")
    
    # Additional information
    st.markdown("---")
    
    # Display input summary
    with st.expander("📋 Input Summary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Day Data:**")
            st.write(f"• Open: ${open_price:.2f}")
            st.write(f"• High: ${high_price:.2f}")
            st.write(f"• Low: ${low_price:.2f}")
            st.write(f"• Volume: {volume:,}")
            st.write(f"• Adj Close: ${adj_close:.2f}")
        
        with col2:
            st.write("**Historical Data:**")
            st.write(f"• Yesterday: ${close_lag1:.2f}")
            st.write(f"• 2 Days Ago: ${close_lag2:.2f}")
            st.write(f"• 3 Days Ago: ${close_lag3:.2f}")
    
    # Model information
    with st.expander("🤖 Model Information"):
        st.write("""
        **Features Used:**
        - OHLCV Data (Open, High, Low, Close, Volume)
        - Technical Indicators (Moving Averages, RSI, Volatility)
        - Price Momentum and Change Metrics
        - Historical Price Lags (1-3 days)
        
        **Model Type:** Machine Learning Ensemble
        
        **Accuracy:** Results may vary based on market conditions
        
        **Disclaimer:** This is for educational purposes only. Not financial advice.
        """)

if __name__ == "__main__":
    main()

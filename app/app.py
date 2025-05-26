import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
import xgboost as XGBRegressor
import streamlit as st
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Function to connect to database and fetch data
def fetch_data_from_db():
    try:
        conn = psycopg2.connect(
            dbname="airflow",
            user="airflow",
            password="airflow",
            host="postgres"
        )
        
        query = """
        SELECT 
            date, gold_close, bitcoin_close, us_index_close, 
            cpi, fed_funds_rate, unemployment_rate
        FROM gold_model_data
        ORDER BY date ASC
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index for time series analysis
        df.set_index('date', inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"Database connection error: {e}")
        # If DB connection fails, use sample data or cached data
        return None

# Function to preprocess data
def preprocess_data(df):
    # Create lag features (previous days' values)
    for i in range(1, 8):
        df[f'gold_lag_{i}'] = df['gold_close'].shift(i)
        df[f'btc_lag_{i}'] = df['bitcoin_close'].shift(i)
        df[f'usd_lag_{i}'] = df['us_index_close'].shift(i)
    
    # Create rolling averages
    df['gold_7d_rolling'] = df['gold_close'].rolling(window=7).mean()
    df['gold_30d_rolling'] = df['gold_close'].rolling(window=30).mean()
    
    # Create percentage changes
    df['gold_pct_change'] = df['gold_close'].pct_change()
    df['btc_pct_change'] = df['bitcoin_close'].pct_change()
    df['usd_pct_change'] = df['us_index_close'].pct_change()
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

# Function to train model
def train_model(df, target_col='gold_close', test_size=0.2, random_state=42):
    # Define features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor.XGBRegressor(random_state=random_state)
    }
    
    best_model = None
    best_score = float('inf')
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
        
        if rmse < best_score:
            best_score = rmse
            best_model = name
    
    return results, best_model, scaler, X_test.index

# Function to make future predictions
def predict_future(model, scaler, last_data, num_days=30):
    future_dates = pd.date_range(start=last_data.index[-1] + timedelta(days=1), periods=num_days, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    # Initialize with last known values
    prev_gold = last_data['gold_close'].iloc[-1]
    prev_btc = last_data['bitcoin_close'].iloc[-1]
    prev_usd = last_data['us_index_close'].iloc[-1]
    prev_cpi = last_data['cpi'].iloc[-1]
    prev_fed = last_data['fed_funds_rate'].iloc[-1]
    prev_unemp = last_data['unemployment_rate'].iloc[-1]
    
    # Create lag features from last data
    gold_lags = [last_data['gold_close'].iloc[-i] if i <= len(last_data) else last_data['gold_close'].iloc[0] for i in range(1, 8)]
    btc_lags = [last_data['bitcoin_close'].iloc[-i] if i <= len(last_data) else last_data['bitcoin_close'].iloc[0] for i in range(1, 8)]
    usd_lags = [last_data['us_index_close'].iloc[-i] if i <= len(last_data) else last_data['us_index_close'].iloc[0] for i in range(1, 8)]
    
    predictions = []
    
    for i in range(num_days):
        # Create a feature row for prediction
        features = {
            'bitcoin_close': prev_btc,
            'us_index_close': prev_usd,
            'cpi': prev_cpi,
            'fed_funds_rate': prev_fed,
            'unemployment_rate': prev_unemp
        }
        
        # Add lag features
        for j in range(1, 8):
            features[f'gold_lag_{j}'] = gold_lags[j-1]
            features[f'btc_lag_{j}'] = btc_lags[j-1]
            features[f'usd_lag_{j}'] = usd_lags[j-1]
        
        # Add rolling averages (simple approximation)
        features['gold_7d_rolling'] = sum(gold_lags[:7]) / 7
        features['gold_30d_rolling'] = prev_gold  # Simplified
        
        # Add percentage changes
        last_gold = gold_lags[0]
        last_btc = btc_lags[0]
        last_usd = usd_lags[0]
        
        features['gold_pct_change'] = (prev_gold - last_gold) / last_gold if last_gold else 0
        features['btc_pct_change'] = (prev_btc - last_btc) / last_btc if last_btc else 0
        features['usd_pct_change'] = (prev_usd - last_usd) / last_usd if last_usd else 0
        
        # Scale features
        X_pred = pd.DataFrame([features])
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        gold_pred = model.predict(X_pred_scaled)[0]
        predictions.append(gold_pred)
        
        # Update for next iteration
        gold_lags = [gold_pred] + gold_lags[:-1]
        btc_lags = [prev_btc] + btc_lags[:-1]
        usd_lags = [prev_usd] + usd_lags[:-1]
        prev_gold = gold_pred
    
    future_df['gold_predicted'] = predictions
    return future_df

# Save model and scaler
def save_model(model, scaler, model_name):
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, f'models/{model_name}_model.pkl')
    joblib.dump(scaler, f'models/{model_name}_scaler.pkl')
    
    st.success(f"Model saved successfully as models/{model_name}_model.pkl")

# Load model and scaler
def load_model(model_name):
    try:
        model = joblib.load(f'models/{model_name}_model.pkl')
        scaler = joblib.load(f'models/{model_name}_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

# Streamlit UI
def main():
    st.title("🏆 Gold Price Prediction Dashboard")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Model Training", "Predictions"])
    
    # Get data
    df = fetch_data_from_db()
    
    if df is None:
        st.error("Unable to fetch data from database. Please check your connection.")
        return
    
    if page == "Data Exploration":
        st.header("Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Data")
            st.dataframe(df.head())
        
        with col2:
            st.subheader("Data Description")
            st.dataframe(df.describe())
        
        st.subheader("Gold Price Over Time")
        fig = px.line(df.reset_index(), x='date', y='gold_close', title='Gold Price Historical Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation with Bitcoin")
            fig = px.scatter(df.reset_index(), x='bitcoin_close', y='gold_close', 
                            title='Gold vs Bitcoin', trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Correlation with US Dollar Index")
            fig = px.scatter(df.reset_index(), x='us_index_close', y='gold_close', 
                            title='Gold vs US Dollar Index', trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Model Training":
        st.header("Model Training")
        
        st.info("Preprocessing data and creating features...")
        processed_df = preprocess_data(df)
        
        st.subheader("Processed Data Preview")
        st.dataframe(processed_df.head())
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a moment."):
                results, best_model_name, scaler, test_dates = train_model(processed_df)
                
                st.success(f"Model training complete! Best model: {best_model_name}")
                
                st.subheader("Model Performance Comparison")
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[m]['rmse'] for m in results],
                    'MAE': [results[m]['mae'] for m in results],
                    'R²': [results[m]['r2'] for m in results]
                })
                
                st.dataframe(metrics_df)
                
                st.subheader("Test Set Predictions")
                fig = go.Figure()
                
                for model_name in results:
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=results[model_name]['predictions'],
                        mode='lines',
                        name=f"{model_name} Prediction"
                    ))
                
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=results[model_name]['actual'],
                    mode='lines',
                    name="Actual Price",
                    line=dict(color='black', dash='dash')
                ))
                
                fig.update_layout(title='Model Predictions vs Actual Prices (Test Set)',
                                xaxis_title='Date',
                                yaxis_title='Gold Price')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save the best model
                for model_name in results:
                    save_model(results[model_name]['model'], scaler, model_name.replace(' ', '_').lower())
        
    elif page == "Predictions":
        st.header("Gold Price Predictions")
        
        model_type = st.selectbox(
            "Select Model", 
            ["random_forest", "gradient_boosting", "xgboost"]
        )
        
        days = st.slider("Number of days to predict", 7, 365, 30)
        
        model, scaler = load_model(model_type)
        
        if model is None:
            st.warning(f"No saved model found for {model_type}. Please train the model first.")
        else:
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    processed_df = preprocess_data(df)
                    future_predictions = predict_future(model, scaler, processed_df, days)
                    
                    # Prepare data for visualization
                    historical = df[['gold_close']].iloc[-180:]  # Last 180 days
                    historical = historical.reset_index()
                    
                    future = future_predictions.reset_index()
                    future.columns = ['date', 'gold_close']
                    
                    # Create plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical['date'],
                        y=historical['gold_close'],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future['date'],
                        y=future['gold_close'],
                        mode='lines',
                        name='Predicted Price',
                        line=dict(color='red')
                    ))
                    
                    fig.add_shape(
                        type="line",
                        x0=historical['date'].iloc[-1],
                        y0=min(historical['gold_close'].min(), future['gold_close'].min()),
                        x1=historical['date'].iloc[-1],
                        y1=max(historical['gold_close'].max(), future['gold_close'].max()),
                        line=dict(color="green", width=2, dash="dash")
                    )
                    
                    fig.update_layout(
                        title=f'Gold Price Prediction for the Next {days} Days',
                        xaxis_title='Date',
                        yaxis_title='Gold Price',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Price Predictions Table")
                    st.dataframe(future)
                    
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=future.to_csv(index=False).encode('utf-8'),
                        file_name=f'gold_predictions_{days}days.csv',
                        mime='text/csv',
                    )

if __name__ == "__main__":
    main()
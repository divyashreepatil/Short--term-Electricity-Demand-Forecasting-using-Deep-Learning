"""
Streamlit App for Short-term Electricity Demand Forecasting
Deployable on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_data(uploaded_file):
    """Load data from uploaded Excel file"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def preprocess_data(data, target_col, seq_len=24):
    """Preprocess data for time series forecasting"""
    # Handle date/time columns
    if 'Date' in data.columns or 'Time' in data.columns:
        dt_col = 'Date' if 'Date' in data.columns else 'Time'
        try:
            data[dt_col] = pd.to_datetime(data[dt_col])
            data = data.sort_values(dt_col).set_index(dt_col)
        except:
            pass
    
    # Handle missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Select features
    features = [target_col]
    dataset = data[features].values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, 0])
    
    return np.array(X), np.array(y), scaler, data


def build_lstm_model(seq_len, n_features):
    """Build LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_rnn_model(seq_len, n_features):
    """Build RNN model"""
    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def plot_training_history(history):
    """Plot training history using Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='#2196F3', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='#FF9800', width=2)
    ))
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epochs',
        yaxis_title='Loss (MSE)',
        template='plotly_white',
        height=400
    )
    return fig


def plot_predictions(y_test, y_pred, scaler, n_features):
    """Plot actual vs predicted using Plotly"""
    # Inverse transform
    y_test_inv = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), 
                       np.zeros((len(y_test), n_features - 1))), axis=1)
    )[:, 0]
    
    y_pred_inv = scaler.inverse_transform(
        np.concatenate((y_pred, 
                       np.zeros((len(y_pred), n_features - 1))), axis=1)
    )[:, 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_test_inv,
        mode='lines',
        name='Actual Demand',
        line=dict(color='#4CAF50', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred_inv,
        mode='lines',
        name='Predicted Demand',
        line=dict(color='#F44336', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Electricity Demand: Actual vs Predicted',
        xaxis_title='Time Steps',
        yaxis_title='Electricity Load',
        template='plotly_white',
        height=450
    )
    return fig, y_test_inv, y_pred_inv


# Main App
def main():
    # Header
    st.title("⚡ Short-term Electricity Demand Forecasting")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model Selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["LSTM", "RNN"],
        help="Choose between LSTM and RNN architectures"
    )
    
    # Parameters
    st.sidebar.subheader("Model Parameters")
    seq_len = st.sidebar.slider(
        "Sequence Length (hours)",
        min_value=6,
        max_value=72,
        value=24,
        help="Number of past hours to use for prediction"
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=5,
        max_value=100,
        value=50
    )
    
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[16, 32, 64, 128],
        value=32
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your Excel or CSV file",
            type=['xlsx', 'csv'],
            help="Upload a file with electricity demand data"
        )
    
    # Sample data option
    if uploaded_file is None:
        st.info("Please upload a data file. The file should contain electricity load data.")
        st.markdown("""
        ### Expected Data Format
        - Column 'BL' for electricity load/demand
        - Optional: Date/Time column, Solar, Wind, Temperature features
        """)
        return
    
    # Load and preprocess data
    with st.spinner("Loading data..."):
        data = load_data(uploaded_file)
    
    if data is None:
        return
    
    with col2:
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(), height=150)
    
    # Show data info
    st.subheader("📈 Dataset Information")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Rows", data.shape[0])
    with col_info2:
        st.metric("Columns", data.shape[1])
    with col_info3:
        st.metric("Features", len(data.columns))
    
    # Show columns
    st.write("Available columns:", ", ".join(data.columns.tolist()))
    
    # Target column selection
    target_col = st.selectbox(
        "Select Target Column (Electricity Demand)",
        data.columns.tolist(),
        index=data.columns.tolist().index('BL') if 'BL' in data.columns else 0
    )
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        X, y, scaler, processed_data = preprocess_data(data, target_col, seq_len)
    
    st.success(f"Data preprocessed! Shape: X={X.shape}, y={y.shape}")
    
    # Train/Test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    st.markdown("---")
    
    # Training section
    st.subheader("🧠 Model Training")
    
    if st.button("🚀 Train Model", type="primary"):
        # Build model
        with st.spinner(f"Building {model_type} model..."):
            if model_type == "LSTM":
                model = build_lstm_model(seq_len, X.shape[2])
            else:
                model = build_rnn_model(seq_len, X.shape[2])
        
        # Show model summary
        with st.expander("Model Architecture"):
            model.summary(print_fn=lambda x: st.text(x))
        
        # Train model
        with st.spinner(f"Training {model_type} model..."):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
        
        st.success("Training completed!")
        
        # Plot training history
        st.subheader("📉 Training History")
        fig_history = plot_training_history(history)
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Predictions
        st.subheader("🎯 Predictions")
        with st.spinner("Making predictions..."):
            y_pred = model.predict(X_test, verbose=0)
        
        # Plot predictions
        fig_pred, y_test_inv, y_pred_inv = plot_predictions(y_test, y_pred, scaler, X.shape[2])
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Metrics
        st.subheader("📊 Model Performance")
        
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("MAE", f"{mae:.3f}")
        with col_m2:
            st.metric("RMSE", f"{rmse:.3f}")
        with col_m3:
            st.metric("R² Score", f"{r2:.3f}")
        
        # Save model
        model.save(f"{model_type.lower()}_electricity_forecast_model.h5")
        st.success(f"✅ {model_type} Model saved successfully!")
        
        # Download link
        with open(f"{model_type.lower()}_electricity_forecast_model.h5", "rb") as f:
            st.download_button(
                label=f"Download {model_type} Model",
                data=f,
                file_name=f"{model_type.lower()}_electricity_forecast_model.h5",
                mime="application/octet-stream"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Short-term Electricity Demand Forecasting using Deep Learning</p>
        <p>Powered by Streamlit, TensorFlow & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

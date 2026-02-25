# Short-term Electricity Demand Forecasting using Deep Learning

Deep learning–based short-term electricity demand forecasting using LSTM, RNN with renewable energy and weather inputs.

This project builds a deep learning model for short-term electricity demand forecasting using LSTM and RNN architectures. It combines historical load data with solar/wind generation and weather parameters to capture temporal patterns for accurate predictions. The workflow includes data normalization, sequence generation, and train-test splitting to ensure reliable model performance.

## 🚀 Project Overview

This work develops a forecasting system using two deep learning architectures:

- **LSTM** (Long Short-Term Memory)
- **RNN** (Recurrent Neural Network)

The models learn temporal dependencies and nonlinear relationships between input features to generate highly reliable short-term demand predictions.

## 🔑 Input Features

The dataset includes:

| Category           | Examples                            |
| ------------------ | ----------------------------------- |
| Historical Load    | Past electricity consumption values |
| Renewable Inputs   | Solar & wind power generation       |
| Weather Parameters | Temperature, humidity, etc.         |

## 🧠 Tech Stack

| Category             | Tools                                   |
| -------------------- | --------------------------------------- |
| Language             | Python                                  |
| Web App              | Streamlit                               |
| Deep Learning        | TensorFlow / Keras                      |
| Supporting Libraries | NumPy, Pandas, Matplotlib, Scikit-learn, Plotly |
| Models               | LSTM, RNN                               |

## 🚀 Streamlit Deployment

This repository is now **Streamlit deploy ready**! You can run the app locally or deploy to Streamlit Cloud.

### Local Installation

```
bash
# Clone the repository
git clone <repository-url>
cd Short--term-Electricity-Demand-Forecasting-using-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy!

The app will automatically be detected and deployed.

## 📱 App Features

- **Data Upload**: Upload your Excel or CSV data files
- **Model Selection**: Choose between LSTM and RNN architectures
- **Adjustable Parameters**: Modify sequence length, epochs, and batch size
- **Real-time Training**: Train models directly in the browser
- **Interactive Visualizations**: Plotly-based charts for training history and predictions
- **Performance Metrics**: MAE, RMSE, and R² Score display
- **Model Download**: Save trained models for future use

## 🔍 Methodology

1. **Data Preprocessing**: 
   - Normalization/scaling
   - Sequence generation for time-series modeling
   - Train–test split

2. **Model Development**: 
   - Individual training of LSTM, RNN models
   - Optimization of temporal learning and pattern extraction

3. **Evaluation**: 
   - Comparison of prediction performance across architectures using standard metrics (MAE, RMSE, R²)

## 📁 Project Structure

```
Short--term-Electricity-Demand-Forecasting-using-Deep-Learning/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── Short_term_electricity.ipynb  # Original Jupyter notebook
├── synthetic_PLN_dataset.xlsx    # Sample dataset
└── README.md                # This file
```


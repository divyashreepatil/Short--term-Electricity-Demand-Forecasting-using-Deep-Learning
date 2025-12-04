# Short--term-Electricity-Demand-Forecasting-using-Deep-Learning
Deep learning–based short-term electricity demand forecasting using LSTM, RNN with renewable energy and weather inputs.
This project builds a deep learning model for short-term electricity demand forecasting using LSTM, RNN, architectures. 
It combines historical load data with solar/wind generation and weather parameters to capture temporal patterns for accurate predictions. 
The workflow includes data normalization, sequence generation, and train-test splitting to ensure reliable model performance.

🚀Project Overview

This work develops a forecasting system using three deep learning architectures:

LSTM (Long Short-Term Memory)

RNN (Recurrent Neural Network)

The models learn temporal dependencies and nonlinear relationships between input features to generate highly reliable short-term demand predictions.

🔑 Input Features

The dataset includes:
| Category           | Examples                            |
| ------------------ | ----------------------------------- |
| Historical Load    | Past electricity consumption values |
| Renewable Inputs   | Solar & wind power generation       |
| Weather Parameters | Temperature, humidity, etc.         |

🧠 Tech Stack: 
| Category             | Tools                                   |
| -------------------- | --------------------------------------- |
| Language             | Python                                  |
| Deep Learning        | TensorFlow / Keras                      |
| Supporting Libraries | NumPy, Pandas, Matplotlib, Scikit-learn |
| Models               | LSTM, RNN                               |

🔍 Methodology
1.Data Preprocessing: Normalization / scaling , Sequence generation for time-series modeling ,Train–test split

2.Model Development : Individual training of LSTM, RNN models , Optimization of temporal learning and pattern extraction

3.Evaluation : Comparison of prediction performance across architectures using standard metrics (MAE, RMSE,)

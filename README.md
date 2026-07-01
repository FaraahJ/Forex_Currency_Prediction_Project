# Currency_Prediction_Project
This repository showcases the process of coding a Streamlit app to predict future exchange rates for 22 different currencies using a range of different programs:

1. The 📁[Analysis](https://github.com/FaraahJ/Forex_Currency_Prediction_Project/tree/main/Analysis) folder contains the separate Jupyter Notebook files showcasing the code used to visually explore the dataset for three modelling techniques:
- LSTM Modelling
- ARIMA/SARIMA
- Prophet Models

  Evaluation of these models are also calculated using MAE, MAPE and RMSE metrics.

2. The [Excel](https://github.com/FaraahJ/Forex_Currency_Prediction_Project/blob/main/MAE%2C%20RMSE%2C%20MAPE%20Metrics%20-%20Sheet1.csv) file records the evaluation metrics and highlights which model was chosen to best represent each currency

3. The 📁 [Models](https://github.com/FaraahJ/Forex_Currency_Prediction_Project/tree/main/Models) folder contains the saved models for each currency. ❗ Unfortunately the SARIMA models will not be found here as their file sizes were over 25MB.

4. The [VSCode file](https://github.com/FaraahJ/Forex_Currency_Prediction_Project/blob/main/app.py) showcases the development of the Streamlit application including a virtual environment created using Python version 3.10 to install and import packages such as Tensorflow.


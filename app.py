import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import prophet as prophet

sns.set()
plt.style.use("seaborn-v0_8-whitegrid")

st.title("Future Forex Currency Price Prediction Model")

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZEALAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN$': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON$': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN$': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}


def load_data():
    data = pd.read_excel(r"C:\Users\Rayha\Downloads\Foreign_Exchange_Rates (1).xls")
    data = data.dropna()
    data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
    return data

#LSTM models
def make_forecast_australia(forecast_length):
    currency_column_name = 'AUSTRALIA - AUSTRALIAN DOLLAR/US$'
    model = tf.keras.models.load_model(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Australia_model.keras'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_newzealand(forecast_length):
    currency_column_name = 'NEW ZEALAND - NEW ZEALAND DOLLAR/US$'
    model = tf.keras.models.load_model(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\NZ_model.keras'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_india(forecast_length):
    currency_column_name = 'INDIA - INDIAN RUPEE/US$'
    model = tf.keras.models.load_model(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\India_model.keras'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_korea(forecast_length):
    currency_column_name = 'KOREA - WON/US$'
    model = tf.keras.models.load_model(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Korea_model.keras'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

#Prophet models
def make_forecast_euro(forecast_length):
    currency_column_name = 'EURO AREA - EURO/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Euro_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_unitedkingdom(forecast_length):
    currency_column_name = 'UNITED KINGDOM - POUND/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\UK_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_canada(forecast_length):
    currency_column_name = 'CANADA - CANADIAN DOLLAR/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Canada_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_china(forecast_length):
    currency_column_name = 'CHINA - YUAN/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\China_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_mexico(forecast_length):
    currency_column_name = 'MEXICO - PESO/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Mexico_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_southafrica(forecast_length):
    currency_column_name = 'SOUTH AFRICA - RAND/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\SouthAfrica_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_singapore(forecast_length):
    currency_column_name = 'SINGAPORE - DOLLAR/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Singapore_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_denmark(forecast_length):
    currency_column_name = 'DENMARK - KRONE/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Denmark_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_japan(forecast_length):
    currency_column_name = 'JAPAN - YEN/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Japan_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_norway(forecast_length):
    currency_column_name = 'NORWAY - KRONE/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Norway_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_sweden(forecast_length):
    currency_column_name = 'SWEDEN - KRONE/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Sweden_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_switzerland(forecast_length):
    currency_column_name = 'SWITZERLAND - FRANK/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Switzerland_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_thailand(forecast_length):
    currency_column_name = 'THAILAND - BAHT/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Thailand_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

#SARIMA models
def make_forecast_brazil(forecast_length):
    currency_column_name = 'BRAZIL - REAL/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Brazil_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_hongkong(forecast_length):
    currency_column_name = 'HONG KONG - DOLLAR/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\HongKong_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_malaysia(forecast_length):
    currency_column_name = 'MALAYSIA - RINGGIT/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Malaysia_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_srilanka(forecast_length):
    currency_column_name = 'SRI LANKA - RUPEE/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Srilanka_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})

def make_forecast_taiwan(forecast_length):
    currency_column_name = 'TAIWAN - NEW DOLLAR/US$'
    model = joblib.load(('C:\\Users\\Rayha\\OneDrive\\Documents\\Data Science Internship 2026\\PROJECT 1\\Currency Models\\Taiwan_model.pkl'))
    data = load_data()
    last_date = data['Time Serie'].iloc[-1]
    last_value = data[currency_column_name].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    forecast_values = np.linspace(last_value, last_value * (1 + 0.01 * forecast_length), forecast_length)
    return pd.DataFrame({'Time Serie': forecast_dates, currency_column_name: forecast_values})


def make_forecast(option, forecast_length):
    if option == 'AUSTRALIAN DOLLAR':
        return make_forecast_australia(forecast_length)
    elif option == 'NEW ZEALAND DOLLAR':
        return make_forecast_newzealand(forecast_length)
    elif option == 'INDIAN RUPEE':
        return make_forecast_india(forecast_length)
    elif option == 'KOREA - WON/US$':
        return make_forecast_korea(forecast_length)
    elif option == 'EURO':
        return make_forecast_euro(forecast_length)
    elif option == 'UNITED KINGDOM - POUND/US$':
        return make_forecast_unitedkingdom(forecast_length)
    elif option == 'CANADA - DOLLAR/US$':
        return make_forecast_canada(forecast_length)
    elif option == 'CHINA - YUAN/US$':
        return make_forecast_china(forecast_length)
    elif option == 'MEXICO - PESO/US$':
        return make_forecast_mexico(forecast_length)
    elif option == 'SOUTH AFRICA - RAND/US$':
        return make_forecast_southafrica(forecast_length)
    elif option == 'SINGAPORE - DOLLAR/US$':
        return make_forecast_singapore(forecast_length)
    elif option == 'DENMARK - KRONE/US$':
        return make_forecast_denmark(forecast_length)
    elif option == 'JAPANESE YEN$':
        return make_forecast_japan(forecast_length)
    elif option == 'NORWAY - KRONE/US$':
        return make_forecast_norway(forecast_length)
    elif option == 'SWEDEN KRONA':
        return make_forecast_sweden(forecast_length)
    elif option == 'SWISS FRANC':
        return make_forecast_switzerland(forecast_length)
    elif option == 'THAI BAHT':
        return make_forecast_thailand(forecast_length)
    elif option == 'BRAZILIAN REAL':
        return make_forecast_brazil(forecast_length)
    elif option == 'HONG KONG DOLLAR':
        return make_forecast_hongkong(forecast_length)
    elif option == 'MALAYSIAN RINGGIT':
        return make_forecast_malaysia(forecast_length)
    elif option == 'SRILANKAN RUPEE':
        return make_forecast_srilanka(forecast_length)
    elif option == 'NEW TAIWAN DOLLAR':
        return make_forecast_taiwan(forecast_length)
    else:
        st.warning(f"No forecast model available for '{option}'.")
        return pd.DataFrame()

# app features
with st.form(key='user_form'):
    selected_option = st.selectbox('Choose a currency:', list(options.keys()))
    forecast = st.number_input(
        "Enter an integer",  # Label displayed to the user
        min_value=1,         # Minimum value allowed
        max_value=100,       # Maximum value allowed
        value=1,             # Default value
        step=1               # Increment step
    )
    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    forecast_df = make_forecast(selected_option, forecast)
    st.write(forecast_df)
    st.line_chart(forecast_df.set_index('Time Serie'))
    st.dataframe(forecast_df)

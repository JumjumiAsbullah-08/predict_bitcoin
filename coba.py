import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.impute import SimpleImputer

# Fungsi untuk mengambil data historis Bitcoin
def get_bitcoin_history(days):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# Fungsi untuk mendapatkan harga Bitcoin dari hari sebelumnya
def get_live_btc_price_previous_day(date):
    date_time = datetime.combine(date, datetime.min.time())
    previous_date = (date_time - pd.Timedelta(days=1)).timestamp()
    current_date = date_time.timestamp()
    endpoint = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {"vs_currency": "usd", "from": previous_date, "to": current_date}
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        if "prices" in data and len(data["prices"]) > 0:
            live_btc_price = data["prices"][0][1]
            return live_btc_price
        else:
            st.warning("Data harga Bitcoin tidak tersedia untuk tanggal ini.")
            return None
    except Exception as e:
        st.warning("Gagal mengambil data harga Bitcoin.")
        return None

# Fungsi untuk mengonversi harga Bitcoin ke Rupiah
def convert_to_rupiah(price):
    response = requests.get(f"https://open.er-api.com/v6/latest/USD")
    data = response.json()
    exchange_rate = data["rates"]["IDR"]
    rupiah_price = price * exchange_rate
    return rupiah_price

# Fungsi untuk mendapatkan data historis selama 30 hari
def get_coingecko_historical_data(days):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        return prices
    else:
        return []

# Fungsi untuk preprocessing data
def preprocess_data(df):
    # Mengganti '-' dengan NaN dan mengisi nilai NaN dengan 0
    df['Vol'] = df['Vol'].replace('-', np.nan).astype(float).fillna(0)
    df['Change'] = df['Change'].str.replace('%', '').astype(float)

    # Menangani nilai NaN
    imputer = SimpleImputer(strategy='mean')
    df[['Open', 'High', 'Low', 'Vol', 'Change']] = imputer.fit_transform(df[['Open', 'High', 'Low', 'Vol', 'Change']])
    
    return df

# Fungsi untuk melatih model dan mencari parameter terbaik
def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    
    # Parameter grid untuk GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Fungsi untuk visualisasi residuals
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Streamlit app
st.title('Prediksi Harga Bitcoin')
st.warning('Harap isi semua input untuk melakukan prediksi')

data_path = "C:/project-btc/asli.csv"
df = pd.read_csv(data_path)

# Preprocessing
df = preprocess_data(df)

# Select features and target variable
features = ['Open', 'High', 'Low', 'Vol', 'Change']
target = 'Price'
X = df[features]
y = df[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Predict function
def predict_price(open_price, high_price, low_price, volume, change_percentage):
    features = [[open_price, high_price, low_price, volume, change_percentage]]
    prediction = model.predict(features)
    return prediction[0]

# Button to trigger prediction
if st.button('Prediksi'):
    open_price = st.text_input('Harga Pembukaan (USD) *', placeholder="Harga Pembuka (USD)")
    high_price = st.text_input('Harga Tertinggi (USD) *', placeholder="Harga Tertinggi (USD)")
    low_price = st.text_input('Harga Terendah (USD) *', placeholder="Harga Terendah (USD)")
    volume = st.text_input('Volume (24H) *', placeholder="Volume (24H)")
    change_percentage = st.number_input('Perubahan Persentase (%) *')
    prediction_date = st.date_input('Tanggal Prediksi *')

    if open_price and high_price and low_price and volume and change_percentage and prediction_date:
        # Convert inputs to float
        open_price = float(open_price)
        high_price = float(high_price)
        low_price = float(low_price)
        volume = float(volume)
        change_percentage = float(change_percentage)

        # Predict the price
        prediction = predict_price(open_price, high_price, low_price, volume, change_percentage)

        # Get the actual price from the previous day
        actual_price = get_live_btc_price_previous_day(prediction_date)

        if actual_price is not None:
            # Calculate MAPE
            mape = (abs(actual_price - prediction) / actual_price) * 100

            # Convert price to Rupiah
            rupiah_prediction = convert_to_rupiah(prediction)
            st.success(f'Prediksi Harga Bitcoin pada tanggal {prediction_date}: $ {prediction:.2f} USD | Rp. {rupiah_prediction:,.2f}', icon="✅")

            # Get live BTC price for the previous day
            live_btc_price_previous_day = get_live_btc_price_previous_day(prediction_date)

            if live_btc_price_previous_day is not None:
                # Convert the live price to Rupiah
                latest_price_btc = convert_to_rupiah(live_btc_price_previous_day)

                # Get historical data for the last 30 days
                historical_data = get_coingecko_historical_data(30)
                if historical_data:
                    x_live = [datetime.utcfromtimestamp(data[0] / 1000).date() for data in historical_data]
                    y_live = [data[1] for data in historical_data]

                    # Create a DataFrame for live data
                    combined_data = {
                        'Tanggal': x_live + [prediction_date],
                        'Harga': y_live + [prediction],
                        'Tipe Data': ['Harga Live BTC'] * len(x_live) + ['Prediksi']
                    }

                    df_combined = pd.DataFrame(combined_data)

                    # Create a Plotly Express area chart for combined data
                    fig = px.line(df_combined, x='Tanggal', y='Harga', markers=True, color='Tipe Data', title='Harga Live Bitcoin (BTC) vs Hasil Prediksi')

                    fig.update_traces(marker=dict(size=10, symbol='circle', color='blue'), selector=dict(name='Prediksi'))
                    fig.add_trace(
                        go.Scatter(
                            x=[prediction_date],
                            y=[prediction],
                            mode='markers+text',
                            marker=dict(size=20, color='red'),
                            text=[f"${prediction:.2f} USD"],
                            textposition='top center',
                            name='Hasil Prediksi'
                        )
                    )

                    fig.update_xaxes(title_text='Tanggal')
                    fig.update_yaxes(title_text='Harga (USD)')
                    st.plotly_chart(fig)

                    # Calculate RMSE
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # Calculate R-squared (R²)
                    r2 = r2_score(y_test, y_pred)

                    with st.expander(f'Mean Absolute Percentage Error (MAPE): {mape:.2f} %'):
                        st.write(f'Prediksi harga Bitcoin pada {prediction_date}: $ {prediction:.2f} USD')
                        st.write(f'Nilai tukar saat ini: Rp {latest_price_btc:,.2f}')
                        st.write(f'RMSE pada data test: {rmse:.2f}')
                        st.write(f'R² (R-squared): {r2:.2f}')

                    # Plot residuals
                    plot_residuals(y_test, y_pred)
                else:
                    st.warning("Data historis harga Bitcoin tidak tersedia.")
        else:
            st.warning("Gagal mendapatkan harga Bitcoin untuk tanggal yang dipilih.")

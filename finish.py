import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import scipy.stats as stats
import pytz
import locale
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # Ganti 'en_US.UTF-8' sesuai dengan pengaturan lokal yang sesuai
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date

favicon_path = 'assets/bitcoin.gif'
st.set_page_config(page_title="Prediksi Harga Bitcoin (BTC)", page_icon=favicon_path)

# untuk halaman css
hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
with st.sidebar:
    # Menampilkan gambar di sidebar
    st.image("assets/bitcoin.gif")

    # Opsi menu
    menu_options = ["Dashboard", "Data", "Prediksi", "Tentang"]
    selected_menu = st.sidebar.selectbox("Pilih Menu", menu_options)
if selected_menu == "Dashboard":
    # Konten untuk Dashboard
    # Membuat judul
    st.title("Live Harga Bitcoin")

    # Fungsi untuk mengambil data historis Bitcoin dari API CoinGecko
    def get_bitcoin_history(days):
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None

    # Mengambil data historis Bitcoin selama 7 hari terakhir
    days = 7
    bitcoin_data = get_bitcoin_history(days)

    # Memproses data jika berhasil diambil
    if bitcoin_data:
        # Membuat DataFrame dari data OHLC (Open, High, Low, Close)
        bitcoin_df = pd.DataFrame(bitcoin_data, columns=["timestamp", "Open", "High", "Low", "Close"])
        
        # Mengubah timestamp menjadi format tanggal
        bitcoin_df["Date"] = pd.to_datetime(bitcoin_df["timestamp"], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        bitcoin_df.drop(columns=["timestamp"], inplace=True)
        
        # Membuat grafik candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=bitcoin_df["Date"],
            open=bitcoin_df["Open"],
            high=bitcoin_df["High"],
            low=bitcoin_df["Low"],
            close=bitcoin_df["Close"]
        )])
        
        # Mengatur judul dan layout grafik
        fig.update_layout(
            title="Live Harga Bitcoin Selama 7 Hari Terakhir (Per 24 Jam)",
            xaxis_title="Tanggal",
            yaxis_title="Harga (USD)",
            xaxis_rangeslider_visible=False
        )
        
        # Menampilkan grafik di Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Gagal mengambil data historis Bitcoin.")
    # Background and Problem Research
    st.title("Apa Itu Bitcoin ? (BTC)")
    # st.write("Latar Belakang :")
    st.markdown(
        """
        <div style="text-align: justify;">
        Bitcoin adalah uang internet sukses pertama yang berbasis teknologi peer-to-peer, di mana tidak ada bank sentral ataupun otoritas yang terlibat dalam transaksi dan produksi mata uang Bitcoin. Mata uang ini diciptakan oleh individu/kelompok anonomim dengan nama Satoshi Nakamoto.
        <br>Kode sumber tersedia secara publik sebagai proyek sumber terbuka, siapa pun dapat melihatnya dan menjadi bagian dari proses pengembangan. Bicotin mengubah cara kita melihat uang sebagaimana kita bicara. Idenya adalah menghasilkan alat pertukaran, yang tidak terikat dengan otoritas bank sentral mana pun, yang dapat ditransfer secara elektronik dengan cara yang aman, dapat diverifikasi, dan tidak dapat diubah.
        <br>Bitcoin adalah mta uang internet peer-to-peer terdesentralisasi yang membuat pembayaran seluler menjadi mudah, biaya transaksi sangat rendah, melindungi identitas Anda, dan berfungsi di mana pun sepanjang waktu tanpa otoritas pusat dan bank.
        <br>Bictoin dirancang untuk hanya memiliki 21 juta BTC yang pernah diciptakan, sehingga membuat BTC sebagai mata uang deflasi. Bitcoin menggunakan alrgoritma hash SHA-256 dengan waktu konfirmasi transaksi rata-rata 10 menit. Hari ini, para penambang menambang Bitcoin menggunakan chip ASIC yang dikhususkan hanya untuk menambang Bitcoin, dan tingkat hash-nya telah melonjak hingga petahash.
        <br>Menjadi mata uang kriptografi online pertama yang sukses, Bitcoin telah menginspirasi mata uang alternatif lain seperti Litecoin, Ethereum, dan lainnya.
        </div>
        """, unsafe_allow_html=True
    )
elif selected_menu == "Data":
    # konten prediksi
    st.title("Data Set dan Algoritma Random Forest")
    st.markdown(
        """
            <h3>Data Set</h3>
            <p style="text-align:justify;">
            Dataset diambil dari platform <a href="https://www.investing.com/crypto/bitcoin/historical-data/">Investing.com</a> mencakup berbagai informasi penting mengenai harga aset kripto Bitcoin (BTC). Data ini mencakup periode waktu tertentu dan berisi Tujuh kolom yang relevan:
            </p>
            <ol style="text-align:justify;">
                <li>Date (Tanggal): Kolom ini mencatat tanggal observasi atau periode waktu tertentu dalam format tanggal. Informasi tanggal ini digunakan untuk menyusun data dalam urutan kronologis.</li>
                <li>Price (Harga): Kolom ini mencatat harga penutupan Bitcoin (BTC) pada tanggal tertentu. Harga penutupan merupakan harga terakhir pada periode perdagangan tersebut dan digunakan sebagai acuan untuk analisis.</li>
                <li>Open (Harga Pembukaan): Kolom ini mencatat harga pembukaan Bitcoin (BTC) pada tanggal tertentu. Harga pembukaan adalah harga pertama yang terjadi pada periode perdagangan tersebut.</li>
                <li>High (Harga Tertinggi): Kolom ini mencatat harga tertinggi yang dicapai oleh Bitcoin (BTC) selama periode perdagangan pada tanggal tertentu. Informasi ini memberikan gambaran mengenai volatilitas harga aset selama periode tersebut.</li>
                <li>Low (Harga Terendah): Kolom ini mencatat harga terendah yang dicapai oleh Bitcoin (BTC) selama periode perdagangan pada tanggal tertentu. Ini menunjukkan tingkat harga minimum yang dicapai oleh instrumen tersebut selama periode tersebut.</li>
                <li>Vol (Volume Perdagangan): Kolom ini mencatat volume perdagangan Bitcoin (BTC) pada tanggal tertentu. Volume perdagangan mencerminkan seberapa besar aktivitas perdagangan yang terjadi pada periode tersebut.</li>
                <li>Change (Persentase Perubahan): Kolom ini mencatat persentase perubahan harga Bitcoin (BTC) dari harga pembukaan hingga harga penutupan pada tanggal tertentu. Informasi ini memberikan gambaran tentang seberapa besar pergerakan harga selama periode perdagangan tersebut.</li>
            
            </ol>
            <p style="text-align:justify;">
            Jumlah data yang akan diambil yaitu 1676 data sample dari Tanggal 01 Januari 2020 s/d 01 Agustus 2024. Berikut Tabel sampling dataset :
            """, unsafe_allow_html=True
    )
    # Membaca dataset dari file CSV
    dataset_path = r"data-set.xlsx"
    df = pd.read_excel(dataset_path)
    
    # Menggantikan indeks DataFrame dengan nomor dari 1 hingga 1992
    df.index = range(1, len(df) + 1)
    
    # Menampilkan dataset
    st.write(df)
    st.markdown(
        """
            <p style="text-align:justify;">
            Dataset ini berharga untuk analisis dan prediksi harga aset kripto. Dengan memanfaatkan informasi tanggal, harga pembukaan, harga penutupan, harga tertinggi, volume perdagangan, dan persentase perubahan, dapat dilakukan berbagai analisis, termasuk analisis tren harga, volatilitas, dan pola perdagangan. Selain itu, dataset ini juga dapat digunakan untuk melatih model prediksi harga menggunakan teknik seperti algoritma Random Forest. Dengan memahami dan menganalisis data ini, dapat diperoleh wawasan berharga tentang dinamika pasar aset kripto Bitcoin (BTC).
            </p>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
            <hr>
            <h3>Algoritma Random Forest</h3>
            <p style="text-align:justify;">
            Random Forest adalah sebuah algoritma pembelajaran mesin yang termasuk dalam kategori Ensemble Learning. Algoritma ini menggunakan konsep penggabungan beberapa pohon keputusan (decision tree) independen untuk melakukan proses klasifikasi atau regresi. Setiap pohon keputusan dalam Random Forest dilatih menggunakan subset acak dari data pelatihan, dan hasil akhir dari algoritma ini didapatkan melalui penggabungan atau voting dari keputusan yang dihasilkan oleh setiap pohon keputusan. Dengan memanfaatkan ansambel dari pohon-pohon keputusan, Random Forest dapat menghasilkan prediksi yang lebih akurat dan memiliki kemampuan untuk menangani kompleksitas data yang tinggi.
            </p>
            <p>Dibawah ini merupakan Pohon Keputusan (Decision Tree) dari Algoritma Random Forest</p>
            """, unsafe_allow_html=True
    )  
    st.image("assets/random forest.jpg")
    st.markdown(
        """
            <p style="text-align:justify;">Pohon keputusan individu memberikan suara untuk hasil kelas dalam contoh mainan random forest. (A) Dataset masukan ini menggambarkan tiga sampel, di mana lima fitur (x1, x2, x3, x4, dan x5) menjelaskan setiap sampel. (B) Pohon keputusan terdiri dari cabang yang bercabang pada titik keputusan. Setiap titik keputusan memiliki aturan yang menentukan apakah sampel akan masuk ke cabang satu atau cabang lain tergantung pada nilai fitur. Cabang-cabang tersebut berakhir pada daun yang termasuk dalam kelas merah atau kelas kuning. Pohon keputusan ini mengklasifikasikan sampel 1 ke kelas merah. (C) Pohon keputusan lainnya, dengan aturan yang berbeda pada setiap titik keputusan. Pohon ini juga mengklasifikasikan sampel 1 ke kelas merah. (D) Random forest menggabungkan suara dari pohon keputusan konstituennya, yang menghasilkan prediksi kelas akhir. (E) Prediksi output akhirnya juga adalah kelas merah (Denisko and Hoffman, 2018).</p>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <hr>
        <h3>Pengujian dan Pengukuran Sistem</h3>
        <p style="text-align:justify;">Dalam setiap pengujian dan pengukuran sistem dapat menggunakan metrik evaluasi, yaitu Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE). Root Mean Squared Error (RMSE) merupakan ukuran besarnya kesalahan hasil prediksi, di mana semakin mendekati 0, nilai RMSE menunjukkan bahwa hasil prediksi semakin akurat. Dengan menggunakan persamaan tersebut, dapat menghitung nilai RMSE untuk mengevaluasi akurasi model prediksi. Semakin kecil nilai RMSE, semakin dekat prediksi model dengan nilai aktual, sehingga menunjukkan performa model yang lebih baik.</p>
        <span>Rumus RMSE (Root Mean Square Error) :</span>
        """, unsafe_allow_html=True
    )
    st.markdown(r'$$\text{RMSE} = \sqrt{ \sum \frac{(X_i - \hat{Y}_i)}{n}^2}$$')
    st.markdown(
        """
        <p style="text-align:justify;">Mean Absolute Percentage Error (MAPE) adalah metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan relatif dari prediksi model dalam bentuk persentase. MAPE menghitung persentase rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya, dibagi dengan nilai sebenarnya, dan kemudian diambil rata-ratanya.</p>
        <span>Rumus MAPE (Mean Absolute Percentage Error) :</span>
        """, unsafe_allow_html=True
    )
    st.markdown(r'$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_i -  ŷ_i}{ŷ_i} \right| x 100\%$$')
elif selected_menu == "Prediksi":
    data_path = "asli.csv"
    df = pd.read_csv(data_path)

    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Pra-pemrosesan data: Mengubah format 'Vol' menjadi float
    df['Vol'] = df['Vol'].replace('-', float('nan'))  # Ganti '-' dengan NaN
    df['Vol'] = df['Vol'].fillna(0)  # Isi nilai NaN dengan 0
    df['Vol'] = df['Vol'].astype(float)  # Konversi ke float

    # Pra-pemrosesan data: Mengubah format 'Change' menjadi float
    df['Change'] = df['Change'].str.replace('%', '').astype(float)

    # Select features and target variable
    features = ['Open', 'High', 'Low', 'Vol', 'Change']
    target = 'Price'

    # Split the data into train and test sets
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict function
    def predict_price(open_price, high_price, low_price, volume, change_percentage):
        features = [[open_price, high_price, low_price, volume, change_percentage]]
        prediction = model.predict(features)
        return prediction[0]

    def get_live_btc_price_previous_day(date):
        # Mengubah date (yang awalnya adalah objek datetime.date) menjadi datetime.datetime
        date_time = datetime.combine(date, datetime.min.time())

        previous_date = (date_time - pd.Timedelta(days=1)).timestamp()
        current_date = date_time.timestamp()

        endpoint = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": previous_date,
            "to": current_date,
        }

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
    # Function to get live BTC price from CoinGecko
    def get_live_btc_price(date):
        endpoint = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",  # Retrieve data for the past 1 day
        }

        try:
            response = requests.get(endpoint, params=params)
            data = response.json()
            if "prices" in data and len(data["prices"]) > 0:
                # Get the most recent price
                current_price = data["prices"][-1][1]
                return current_price
            else:
                st.warning("Data harga Bitcoin tidak tersedia.")
                return None
        except Exception as e:
            st.warning("Gagal mengambil data harga Bitcoin.")
            return None
            
    # Function to convert Bitcoin price to Rupiah
    def convert_to_rupiah(price):
        response = requests.get(f"https://open.er-api.com/v6/latest/USD")
        data = response.json()
        exchange_rate = data["rates"]["IDR"]
        rupiah_price = price * exchange_rate
        return rupiah_price

    # Function to get historical data for 30 days
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

    # Streamlit app
    st.title('Prediksi Harga Bitcoin')
    st.warning('Harap isi semua input untuk melakukan prediksi')

    # Create two columns for input features
    col1, col2 = st.columns(2)

    # Input features for prediction in the first column
    with col1:
        open_price = st.text_input('Harga Pembukaan (USD) *', placeholder="Harga Pembuka (USD)")
        high_price = st.text_input('Harga Tertinggi (USD) *', placeholder="Harga Tertinggi (USD)")
        low_price = st.text_input('Harga Terendah (USD) *', placeholder="Harga Terendah (USD)")

    # Input features for prediction in the second column
    with col2:
        volume = st.text_input('Volume (24H) *', placeholder="Volume (24H)")
        change_percentage = st.number_input('Perubahan Persentase (%) *')
        prediction_date = st.date_input('Tanggal Prediksi *')

    # Button to trigger prediction
    if st.button('Prediksi'):
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
            current_price = get_live_btc_price(datetime.today())

            if actual_price is not None:
                # Calculate MAPE
                mape = (abs(actual_price - prediction) / actual_price) * 100

                # Convert price to Rupiah
                rupiah_prediction = convert_to_rupiah(prediction)
                st.success(f'Prediksi Harga Bitcoin pada tanggal {prediction_date}: $ {prediction:.2f} USD | Rp. {rupiah_prediction:,.2f}', icon="✅")
                # Get live BTC price for today
                rupiah_current_price = convert_to_rupiah(current_price)
                if current_price is not None:
                    rupiah_current_price = convert_to_rupiah(current_price)
                    st.info(f'Harga Bitcoin pada tanggal hari ini [{datetime.today().strftime("%Y-%m-%d")}]: $ {current_price:.2f} USD | Rp. {rupiah_current_price:,.2f}', icon="ℹ")

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
                        live_data = {
                            'Tanggal': x_live,
                            'Harga': y_live,
                            'Tipe Data': ['Harga Live BTC'] * len(x_live)
                        }

                        # Combine the live data and prediction data into one DataFrame
                        combined_data = {
                            'Tanggal': x_live + [prediction_date],
                            'Harga': y_live + [prediction],
                            'Tipe Data': ['Harga Live BTC'] * len(x_live) + ['Prediksi']
                        }

                        df_combined = pd.DataFrame(combined_data)

                        # Create a Plotly Express area chart for combined data
                        fig = px.line(df_combined, x='Tanggal', y='Harga', markers=True, color='Tipe Data', title='Harga Live Bitcoin (BTC) vs Hasil Prediksi')

                        # Customize the layout for markers
                        fig.update_traces(marker=dict(size=10, symbol='circle', color='blue'), selector=dict(name='Prediksi'))

                        # Add a large marker with a label for the prediction point
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

                        # Update the axes titles
                        fig.update_xaxes(title_text='Tanggal')
                        fig.update_yaxes(title_text='Harga (USD)')

                        # Show the combined data chart in Streamlit
                        st.plotly_chart(fig)

                        # Menghitung MAPE dan R²
                        y_pred = model.predict(X_test)

                        # MAPE
                        # mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                        # R²
                        r2 = r2_score(y_test, y_pred)

                        with st.expander(f'Mean Absolute Percentage Error (MAPE): {mape:.2f} %'):
                            st.write(f"MAPE adalah metrik untuk mengukur tingkat kesalahan dalam prediksi perbandingan persentase antara prediksi dan data aktual. Nilai MAPE yang lebih rendah menunjukkan prediksi yang lebih akurat. Hasil MAPE: {mape:.2f} %")

                        with st.expander(f'R-squared (R²): {r2:.2f}'):
                            st.write(f"R-squared (R²) mengukur seberapa baik model Anda menjelaskan variasi dalam data. Nilai R² yang lebih tinggi menunjukkan model yang lebih baik. Hasil R²: {r2:.2f}")

                        # Creating bar chart for MAPE and R²
                        fig_bar = px.bar(x=['MAPE', 'R²'],
                                        y=[mape, r2],
                                        color=['MAPE', 'R²'],
                                        labels={'y': 'Nilai'},
                                        title='Grafik Batang MAPE dan R²',
                                        barmode='group',
                                        width=600)

                        # Display bar chart in Streamlit
                        st.plotly_chart(fig_bar)

                        # Load dataset from CSV file
                        file_path = 'asli.csv'
                        df = pd.read_csv(file_path)

                        # Preprocessing
                        df['Vol'] = df['Vol'].replace('-', float('nan'))  # Replace '-' with NaN
                        df['Vol'] = df['Vol'].fillna(0)  # Fill NaN with 0
                        df['Change'] = df['Change'].str.replace('%', '').astype(float)  # Convert 'Change' to float

                        # Select features and target variable
                        features = ['Date', 'Open', 'High', 'Low', 'Vol', 'Change']
                        target = 'Price'

                        # Split the data into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

                        # Initialize and train the Random Forest model
                        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        model_rf.fit(X_train.drop(columns=['Date']), y_train)  # Drop 'Date' for training

                        # Predict on the training and testing sets
                        y_train_pred = model_rf.predict(X_train.drop(columns=['Date']))
                        y_test_pred = model_rf.predict(X_test.drop(columns=['Date']))

                        # Create a dataframe for the results of training and testing
                        result_train_df = pd.DataFrame({
                            'Date': X_train['Date'],
                            'Hasil Testing/Training': y_train.values - y_train_pred,
                            'Set': 'Training'
                        })

                        result_test_df = pd.DataFrame({
                            'Date': X_test['Date'],
                            'Hasil Testing/Training': y_test.values - y_test_pred,
                            'Set': 'Testing'
                        })

                        # Combine the results into one dataframe
                        result_combined_df = pd.concat([result_train_df, result_test_df], ignore_index=True)

                        # Streamlit app
                        # st.title('Hasil Training dan Testing Random Forest')

                        # Plot hasil training
                        st.subheader('Plot Scatter Hasil Training')
                        fig_train, ax_train = plt.subplots(figsize=(10, 5))
                        ax_train.scatter(result_combined_df[result_combined_df['Set'] == 'Training']['Date'], 
                                        result_combined_df[result_combined_df['Set'] == 'Training']['Hasil Testing/Training'], 
                                        color='blue', alpha=0.5, s=50, label='Training')
                        ax_train.set_title('Hasil Training')
                        ax_train.set_xlabel('Date')
                        ax_train.set_ylabel('Hasil Testing/Training')
                        ax_train.legend()
                        st.pyplot(fig_train)

                        # Plot hasil testing
                        st.subheader('Plot Scatter Hasil Testing')
                        fig_test, ax_test = plt.subplots(figsize=(10, 5))
                        ax_test.scatter(result_combined_df[result_combined_df['Set'] == 'Testing']['Date'], 
                                        result_combined_df[result_combined_df['Set'] == 'Testing']['Hasil Testing/Training'], 
                                        color='red', alpha=0.5, s=50, label='Testing')
                        ax_test.set_title('Hasil Testing')
                        ax_test.set_xlabel('Date')
                        ax_test.set_ylabel('Hasil Testing/Training')
                        ax_test.legend()
                        st.pyplot(fig_test)
                    else:
                        st.warning("Gagal mengambil data historis dari API.")
            else:
                st.warning("Gagal mengambil harga live Bitcoin.")
        else:
            st.warning('Harap isi semua input yang diperlukan.')
    # Tombol Reset
    if st.button('Reset'):
        # Setel ulang parameter query
        # st.experimental_set_query_params()

        # Redirect ke halaman yang sama untuk mereset aplikasi
        st.stop()  # Hentikan eksekusi skrip di sini untuk efek reset
elif selected_menu == "Tentang":
   # Membagi layout menjadi 2 kolom
    col1, col2 = st.columns(2)

    # Gambar dan Biodata untuk Pembimbing Artikel Jurnal
    gambar_pembimbing_path = "assets/pak sam.jpg"
    biodata_pembimbing = """
    **Nama :** Samsudin, ST,M.Kom

    **Peneliti :** Prediksi Harga Cryptocurrency Bitcoin (BTC) Dengan Informasi Blokchain Menggunakan Algoritma Machine Learning

    **Publikasi :**
    - ID Sinta : 6003868
    - ID Scopus : 57209425430
    - Google Scholar : https://scholar.google.co.id/citations?user=_QmOWZ4AAAAJ&hl=id
    - ORCID : https://orcid.org/0000-0003-2219-2747 
    """
    col1.image(gambar_pembimbing_path, caption="Researcher", use_column_width=True, width=150)
    col1.markdown("---")  # Garis pembatas
    col2.markdown(biodata_pembimbing)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
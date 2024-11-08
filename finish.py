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
            Jumlah data yang akan diambil yaitu 1675 data sample dari Tanggal 01 Januari 2020 s/d 01 Agustus 2024. Berikut Tabel sampling dataset :
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
        <p style="text-align:justify;">Dalam setiap pengujian dan pengukuran sistem dapat menggunakan metrik evaluasi, yaitu R-squared (R2) dan Mean Absolute Percentage Error (MAPE). R-squared (R2) merupakan ukuran yang menunjukkan seberapa baik model prediksi mampu menjelaskan variasi dalam data aktual. Nilai R2 berkisar antara 0 hingga 1, di mana semakin mendekati 1, nilai R2 menunjukkan bahwa model prediksi semakin akurat. Dengan menggunakan persamaan tersebut, dapat menghitung nilai R2 untuk mengevaluasi performa model prediksi. Semakin tinggi nilai R2, semakin baik model dalam memprediksi nilai aktual.</p>
        <span>Rumus R-squared (R2) :</span>
        """, unsafe_allow_html=True
    )

    st.markdown(r'$$\text{R}^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}$$')
    st.markdown(
        """
        <p style="text-align:justify;">Mean Absolute Percentage Error (MAPE) adalah metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan relatif dari prediksi model dalam bentuk persentase. MAPE menghitung persentase rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya, dibagi dengan nilai sebenarnya, dan kemudian diambil rata-ratanya.</p>
        <span>Rumus MAPE (Mean Absolute Percentage Error) :</span>
        """, unsafe_allow_html=True
    )
    st.markdown(r'$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_i -  ŷ_i}{ŷ_i} \right| x 100\%$$')
elif selected_menu == "Prediksi":
    # Fungsi untuk mendapatkan harga BTC terkini dari API CoinGecko
    def get_live_btc_price():
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
            data = response.json()
            return data['bitcoin']['usd']
        except Exception as e:
            st.warning(f"Gagal mengambil data harga Bitcoin terkini dari CoinGecko: {e}")
            return None

    # Fungsi untuk mendapatkan data historis BTC dari API CoinGecko
    def get_historical_btc_prices(days=30):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            response = requests.get(
                'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range',
                params={
                    'vs_currency': 'usd',
                    'from': int(start_date.timestamp()),
                    'to': int(end_date.timestamp())
                }
            )
            data = response.json()
            prices = data['prices']
            # Mengonversi timestamp ke objek datetime
            prices = [(datetime.fromtimestamp(price[0] / 1000), price[1]) for price in prices]
            return prices
        except Exception as e:
            st.warning(f"Gagal mengambil data historis harga Bitcoin dari CoinGecko: {e}")
            return None

    # Fungsi untuk konversi harga Bitcoin ke Rupiah
    def convert_to_rupiah(price):
        try:
            response = requests.get("https://open.er-api.com/v6/latest/USD")
            data = response.json()
            exchange_rate = data["rates"]["IDR"]
            return price * exchange_rate
        except Exception as e:
            st.warning(f"Gagal mengonversi harga ke Rupiah: {e}")
            return None

    # Baca data dari CSV
    data_path = "asli.csv"
    df = pd.read_csv(data_path)

    # Pra-pemrosesan data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Vol'] = df['Vol'].replace('-', float('nan')).fillna(0).astype(float)
    df['Change'] = df['Change'].str.replace('%', '').astype(float)

    # Pilih fitur dan target
    features = ['Open', 'High', 'Low', 'Vol', 'Change']
    target = 'Price'
    X = df[features]
    y = df[target]

    # Split data untuk train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model Random Forest
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Fungsi untuk konversi string ke float dengan titik desimal
    def convert_to_float(value):
        try:
            value = value.replace(',', '.')
            return float(value)
        except ValueError:
            st.error("Format angka salah. Harap gunakan titik (.) sebagai pemisah desimal, bukan koma (,).")
            return None

    # Validasi input
    def validate_inputs(open_price, high_price, low_price, volume, change_percentage):
        if open_price is None or high_price is None or low_price is None or volume is None:
            st.warning('Harap isi semua input yang diperlukan.')
            return False
        if high_price < low_price:
            st.error('Harga tertinggi tidak boleh lebih rendah dari harga terendah.')
            return False
        return True

    # Fungsi prediksi harga
    def predict_price(open_price, high_price, low_price, volume, change_percentage, model):
        features = [[open_price, high_price, low_price, volume, change_percentage]]
        prediction = model.predict(features)
        return prediction[0]

    # Streamlit app
    st.title('Prediksi Harga Bitcoin')
    st.warning('Harap isi semua input untuk melakukan prediksi')

    # Buat dua kolom untuk input
    col1, col2 = st.columns(2)

    with col1:
        open_price = st.text_input('Harga Pembukaan (USD) *', placeholder="Harga Pembuka (USD)")
        high_price = st.text_input('Harga Tertinggi (USD) *', placeholder="Harga Tertinggi (USD)")
        low_price = st.text_input('Harga Terendah (USD) *', placeholder="Harga Terendah (USD)")

    with col2:
        volume = st.text_input('Volume (24H) *', placeholder="Volume (24H)")
        change_percentage = st.number_input('Perubahan Persentase (%) *')
        prediction_date = st.date_input('Tanggal Prediksi *')

    # Tombol prediksi
    if st.button('Prediksi'):
        if open_price and high_price and low_price and volume and change_percentage and prediction_date:
            # Konversi input ke float
            open_price = convert_to_float(open_price)
            high_price = convert_to_float(high_price)
            low_price = convert_to_float(low_price)
            volume = convert_to_float(volume)
            change_percentage = float(change_percentage)

            if validate_inputs(open_price, high_price, low_price, volume, change_percentage):
                # Prediksi harga
                prediction = predict_price(open_price, high_price, low_price, volume, change_percentage, model)
                rupiah_prediction = convert_to_rupiah(prediction)
                st.success(f'Prediksi Harga Bitcoin pada tanggal {prediction_date.strftime("%Y-%m-%d")}: $ {prediction:.2f} USD | Rp. {rupiah_prediction:,.2f}', icon="✅")

                # Dapatkan harga BTC terkini
                current_price = get_live_btc_price()
                if current_price is not None:
                    rupiah_current_price = convert_to_rupiah(current_price)
                    st.info(f'Harga Bitcoin pada tanggal hari ini [{datetime.today().strftime("%Y-%m-%d")}]: $ {current_price:.2f} USD | Rp. {rupiah_current_price:,.2f}', icon="ℹ")

                # Dapatkan dan tampilkan data historis BTC
                historical_data = get_historical_btc_prices(30)
                if historical_data:
                    x_live = [data[0] for data in historical_data]
                    y_live = [data[1] for data in historical_data]

                    # Buat DataFrame untuk data historis
                    combined_data = {
                        'Tanggal': x_live + [prediction_date],
                        'Harga': y_live + [prediction],
                        'Tipe Data': ['Harga Live BTC'] * len(x_live) + ['Prediksi']
                    }
                    df_combined = pd.DataFrame(combined_data)

                    # Plot garis data historis dan prediksi
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


                # Hitung MAPE dan R²
                y_pred = model.predict(X_test)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)

                with st.expander(f'Mean Absolute Percentage Error (MAPE): {mape:.2f} %'):
                    st.write(f"MAPE adalah metrik untuk mengukur tingkat kesalahan dalam prediksi perbandingan persentase antara prediksi dan data aktual. Nilai MAPE yang lebih rendah menunjukkan prediksi yang lebih akurat. Hasil MAPE: {mape:.2f} %")

                with st.expander(f'R-squared (R²): {r2:.2f}'):
                    st.write(f"R-squared (R²) mengukur seberapa baik model Anda menjelaskan variasi dalam data. Nilai R² yang lebih tinggi menunjukkan model yang lebih baik. Hasil R²: {r2:.2f}")

                # Grafik batang MAPE dan R²
                fig_bar = px.bar(x=['MAPE', 'R²'], y=[mape, r2], color=['MAPE', 'R²'], labels={'y': 'Nilai'}, title='Grafik Batang MAPE dan R²', barmode='group', width=600)
                st.plotly_chart(fig_bar)

                # Grafik scatter hasil training dan testing
                result_train_df = pd.DataFrame({
                    'Date': X_train.index,
                    'Hasil Testing/Training': y_train.values - model.predict(X_train),
                    'Set': 'Training'
                })

                result_test_df = pd.DataFrame({
                    'Date': X_test.index,
                    'Hasil Testing/Training': y_test.values - y_pred,
                    'Set': 'Testing'
                })

                result_combined_df = pd.concat([result_train_df, result_test_df], ignore_index=True)

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
    gambar_pembimbing_path = "assets/profil.jpg"
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
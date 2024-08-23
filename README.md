# Predict Bitcoin Project
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=NumPy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-377EB8?style=flat-square&logo=Matplotlib&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=Plotly&logoColor=white)
![openpyxl](https://img.shields.io/badge/-openpyxl-00ADEF?style=flat-square&logo=openpyxl&logoColor=white)

This repository contains a Streamlit web application for predicting sales based on various features. The prediction model is built using scikit-learn and data manipulation is done using pandas and numpy. Visualization is handled with Altair, Matplotlib, and Plotly. The project also involves reading and writing Excel files using openpyxl.

## Libraries Used

- [Streamlit](https://streamlit.io/): For creating interactive web applications with Python.
- [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- [NumPy](https://numpy.org/): For numerical computing in Python.
- [scikit-learn](https://scikit-learn.org/stable/): For machine learning modeling and predictive analytics.
- [Altair](https://altair-viz.github.io/): For declarative statistical visualization.
- [Matplotlib](https://matplotlib.org/): For creating static, animated, and interactive visualizations in Python.
- [Plotly](https://plotly.com/python/): For interactive and publication-quality graphs.

## Installation
Untuk menjalankan aplikasi ini secara lokal, ikuti langkah-langkah berikut:
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```
2. Buat dan aktifkan lingkungan virtual (opsional tetapi direkomendasikan):
   ```bash
   python -m venv env
   source env/bin/activate  # Di Windows gunakan `env\Scripts\activate`   
   ```
3. Instal dependensi yang tercantum dalam requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
## Cara Menggunakan Aplikasi
1. Unggah Dataset: Aplikasi ini memungkinkan pengguna untuk mengunggah dataset dalam format CSV yang berisi data historis harga Bitcoin dan berbagai fitur lainnya. Pastikan file CSV yang diunggah memiliki kolom-kolom yang sesuai untuk proses prediksi.
2. Eksplorasi Data: Setelah dataset diunggah, aplikasi menyediakan berbagai visualisasi dan statistik deskriptif untuk memahami data yang akan digunakan untuk prediksi. Ini termasuk visualisasi time series, distribusi fitur, dan korelasi antara variabel.
3. Pelatihan Model: Aplikasi menggunakan algoritma Random Forest Regressor dari scikit-learn untuk membangun model prediksi. Pengguna dapat menyesuaikan beberapa parameter model sebelum memulai pelatihan.
4. Evaluasi Model: Setelah model dilatih, aplikasi akan menampilkan metrik evaluasi seperti Mean Absolute Percentage Error (MAPE), dan R-squared (R2) untuk mengukur kinerja model.
5. Prediksi: Pengguna dapat melakukan prediksi harga Bitcoin berdasarkan data baru yang diunggah. Hasil prediksi akan ditampilkan dalam bentuk grafik dan tabel.

## Kontribusi
Jika Anda ingin berkontribusi pada proyek ini, Anda bisa memulai dengan melakukan fork pada repositori ini dan membuat pull request. Anda juga bisa melaporkan masalah atau memberikan saran melalui bagian Issues di GitHub.

## Lisensi
Proyek ini dilisensikan di bawah MIT License. Anda bebas untuk menggunakan, menyalin, memodifikasi, dan mendistribusikan proyek ini sesuai dengan ketentuan lisensi.

## Kredit
1. Proyek ini dibangun menggunakan Streamlit, sebuah framework yang memudahkan pembuatan aplikasi web interaktif dengan Python.
2. Terima kasih kepada Pandas, NumPy, scikit-learn, Matplotlib, dan Plotly yang menyediakan alat-alat penting untuk analisis data dan pembelajaran mesin.
   

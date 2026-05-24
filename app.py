import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==========================================
# 0. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="NOPI - Dashboard Intelijen Bisnis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. LOAD DATASET ASLI (Jalur: data/)
# ==========================================
@st.cache_data
def load_nopi_data():
    base_path = os.path.dirname(__file__)
    
    file_primer = os.path.join(base_path, 'data', 'Dataset_Terstruktur_Primer_NOPI.csv')
    file_eval = os.path.join(base_path, 'data', 'evaluasi_3_model.csv')
    file_detail = os.path.join(base_path, 'data', 'detail_akurasi_semua_model.csv')
    file_komparasi = os.path.join(base_path, 'data', 'hasil_komparasi_ocr_final.csv')
    
    df_primer = pd.read_csv(file_primer)
    df_eval = pd.read_csv(file_eval)
    df_detail = pd.read_csv(file_detail)
    df_komparasi = pd.read_csv(file_komparasi)
    
    # Standarisasi kolom tanggal & pembersihan data dasar
    for col in df_primer.columns:
        if 'tanggal' in col.lower() or 'date' in col.lower():
            df_primer[col] = pd.to_datetime(df_primer[col], errors='coerce')
            df_primer['Tanggal_Clean'] = df_primer[col]
            
    return df_primer, df_eval, df_detail, df_komparasi

# Memuat data
try:
    df_primer, df_eval, df_detail, df_komparasi = load_nopi_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_msg = str(e)

# ==========================================
# 2. SIDEBAR NAVIGASI & FILTER DATA REAL
# ==========================================
st.sidebar.title("🔍 Menu Utama NOPI")
page = st.sidebar.radio("Pilih Panel Dashboard:", 
    ["Analisis Pipeline Sistem", "Performa Model AI", "Insight Data Transaksi"])

st.sidebar.markdown("---")

if data_loaded:
    # Identifikasi kolom barang dan harga secara dinamis untuk pengolahan insight
    col_barang = 'nama_barang' if 'nama_barang' in df_primer.columns else df_primer.columns[0]
    col_harga = 'total_harga' if 'total_harga' in df_primer.columns else df_primer.columns[-1]
    
    # Deteksi kolom kategori harga
    kolom_filter = 'kategori_harga' if 'kategori_harga' in df_primer.columns else None

    # Filter Global di Sidebar (Hanya aktif untuk Panel 3)
    if page == "Insight Data Transaksi":
        st.sidebar.subheader("🎛️ Filter Interaktif Bisnis")
        
        # 1. Filter Kategori/Nama Item
        if kolom_filter:
            opsi_kategori = df_primer[kolom_filter].dropna().unique().tolist()
            selected_cat = st.sidebar.multiselect("Pilih Kategori Harga:", opsi_kategori, default=opsi_kategori)
            df_filtered = df_primer[df_primer[kolom_filter].isin(selected_cat)]
        else:
            opsi_barang = df_primer[col_barang].dropna().unique().tolist()[:15]
            selected_items = st.sidebar.multiselect("Pilih Sampel Produk:", opsi_barang, default=opsi_barang[:5])
            df_filtered = df_primer[df_primer[col_barang].isin(selected_items)]
        
        # 2. Filter Rentang Tanggal
        if 'Tanggal_Clean' in df_primer.columns and not df_primer['Tanggal_Clean'].isna().all():
            min_date = df_primer['Tanggal_Clean'].min().date()
            max_date = df_primer['Tanggal_Clean'].max().date()
            date_range = st.sidebar.date_input("Rentang Tanggal Transaksi:", [min_date, max_date])
            # Filter tanggal
            if len(date_range) == 2:
                df_filtered = df_filtered[
                    (df_filtered['Tanggal_Clean'].dt.date >= date_range[0]) & 
                    (df_filtered['Tanggal_Clean'].dt.date <= date_range[1])
                ]
else:
    st.error(f"Gagal memuat berkas dari folder `data/`. Struktur direktori salah. Deskripsi Error: {error_msg}")
    st.stop()

# ==========================================
# 3. KONTEN DASHBOARD UTAMA
# ==========================================
st.title("🚀 Dashboard Intelijen Bisnis - NOPI (Nota Pintar)")
st.markdown("**Solusi Otomasi Finansial UMKM Berbasis Integrasi Jaringan Sensor Citra CNN & Ekstraksi PaddleOCR**")
st.markdown("---")

# ------------------------------------------
# PANEL 1: ANALISIS PIPELINE SISTEM
# ------------------------------------------
if page == "Analisis Pipeline Sistem":
    st.header("⚙️ Kerangka Kerja Pipeline Sistem Dua Tahap")
    
    # Ringkasan Eksekutif Konsep Bisnis
    with st.expander("📖 Latar Belakang & Pernyataan Masalah Proyek (Klik untuk Membaca)", expanded=True):
        st.markdown("""
        * **Konteks Makro:** Indonesia memiliki 65,5 juta UMKM yang menyumbang 61% terhadap PDB nasional. Namun, pelaku toko sembako mayoritas mengandalkan ingatan atau catatan kertas manual dalam melacak keuangan.
        * **Permasalahan Utama:** Rekapitulasi finansial lambat, rawan kesalahan manusia (*human error*), serta tidak adanya visualisasi laba per item produk secara *real-time*.
        * **Solusi NOPI:** Membangun *pipeline* otomatis hibrida berseri:
          1. **Jaringan CNN Classifier:** Memfilter citra input berskala biner untuk membedakan gambar struk dari gambar non-struk secara otomatis sebelum sistem bekerja lebih jauh.
          2. **Arsitektur OCR:** Mengambil data karakter teks tak terstruktur untuk dipetakan secara spasial ke dalam format tabel basis data terstruktur (Nama Item, Harga Satuan, Total Transaksi).
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🛡️ Tahap 1: Filtrasi Gerbang Jaringan CNN")
        st.info("""
        * **Fungsi:** Mengklasifikasikan secara otomatis gambar masukan pengguna.
        * **Validasi Target:** Memastikan akurasi minimal 85% untuk menolak gambar *non-struk* (wajah, pemandangan, objek acak).
        * **Dampak Bisnis:** Menghilangkan data sampah (*noise data*) di awal rantai komputasi sehingga kinerja server serverless cloud stabil.
        """)
        
    with col2:
        st.subheader("📝 Tahap 2: Ekstraksi Informasi Struktural OCR")
        st.info("""
        * **Fungsi:** Membaca baris per baris teks nota belanja yang lolos klasifikasi Tahap 1.
        * **Komponen yang Diambil:** String teks nama produk, kuantitas penjualan, nilai harga satuan (*unit price*), dan total nominal struk belanja.
        * **Dampak Bisnis:** Konversi instan dari kertas fisik usang menjadi basis data digital siap olah tanpa entri manual.
        """)

    st.markdown("---")
    st.subheader("📋 Sampel Basis Data Hasil Ekstraksi Sukses (`Dataset_Terstruktur_Primer_NOPI.csv`)")
    st.write("Tabel di bawah ini merepresentasikan 10 baris data mentah primer asli hasil pengolahan ekstraksi teks terstruktur:")
    st.dataframe(df_primer.head(10), use_container_width=True)
    
    st.markdown("""
    **💡 Narasi Analisis Alur Kerja:**
    Penerapan arsitektur gabungan dua tahap ini menjamin integritas data transaksi. Pemisahan kerja antara CNN (klasifikasi spasial gambar) dan OCR (pemetaan teks) meminimalkan pemborosan daya komputasi. Hanya gambar dokumen transaksi valid yang akan dibebankan ke dalam modul pemrosesan PaddleOCR.
    """)

# ------------------------------------------
# PANEL 2: PERFORMA MODEL AI
# ------------------------------------------
elif page == "Performa Model AI":
    st.header("📊 Integrasi Metrik & Evaluasi Model Pengenalan Karakter")
    st.write("Analisis performa komparasi arsitektur berdasarkan pengujian data riil instrumen pengujian.")

    # Menampilkan tabel Indikator Pengukuran BQ1
    st.subheader("📌 Indikator Pengukuran Business Question 1 (BQ1)")
    st.markdown("""
    Pengujian model OCR dinilai secara objektif berdasarkan empat metrik utama: *Success Rate* parsing, akurasi jumlah baris item produk, akurasi rekap total nominal harga, dan kecepatan durasi waktu pemrosesan (*inference time*).
    """)

    # Visualisasi Utama Komparasi Model OCR
    col_x = df_eval.columns[0]
    col_y = df_eval.columns[1]
    
    fig_ev = px.bar(
        df_eval, x=col_x, y=col_y, color=col_x,
        title="Grafik 1. Perbandingan Tingkat Keberhasilan Pemrosesan Model OCR (%)",
        labels={col_x: "Nama Model Arsitektur", col_y: "Persentase Nilai (%)"}
    )
    fig_ev.update_layout(xaxis_title="Arsitektur Model OCR", yaxis_title="Tingkat Keberhasilan (Success Rate %)")
    st.plotly_chart(fig_ev, use_container_width=True)

    # Memasukkan Analisis Tertulis dari Data Evaluasi Riil
    st.markdown("""
    #### **📖 Narasi Eksperimen Metrik Akurasi Bisnis (Insight BQ1):**
    Berdasarkan hasil evaluasi komparasi tiga arsitektur model, ditemukan beberapa poin evaluasi kritis:
    * **Model Terpilih (PaddleOCR):** Memiliki performa yang **paling seimbang**. Paddle memperoleh *Success Rate* tertinggi sebesar **73.33%** dan akurasi total harga tertinggi sebesar **26.09%**, meskipun durasi waktu pengolahan komputasinya berada sedikit di bawah Tesseract.
    * **Kelemahan Tesseract:** Memiliki waktu pemrosesan paling cepat (**2.64 detik**), tetapi nilai akurasi jumlah baris item dan total harga adalah yang **paling rendah**, sehingga tidak andal untuk aplikasi pembukuan finansial yang sensitif terhadap kesalahan nominal uang.
    * **Kelemahan EasyOCR:** Menghasilkan akurasi ekstraksi jumlah item tertinggi (**33.5%**), namun waktu pemrosesan komputasinya **sangat lambat (18.73 detik)** serta akurasi total harganya sangat rendah (**14.0%**), yang menyebabkannya tidak efisien untuk kebutuhan operasional harian (*production deployment*).
    
    * **Kesimpulan Finansial:** **PaddleOCR** secara mutlak dipilih sebagai penggerak utama dalam proyek aplikasi NOPI karena memprioritaskan stabilitas pembacaan nilai angka total harga belanja konsumen dengan performa durasi waktu yang wajar.
    """)
    
    st.markdown("---")
    st.subheader("📋 Tabel Metrik Komparasi Detak Data (`hasil_komparasi_ocr_final.csv`)")
    st.dataframe(df_komparasi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Log Rincian Akurasi Berkas Uji (`detail_akurasi_semua_model.csv`)")
    st.write("Berikut adalah 15 baris log data sampel hasil pengujian detail akurasi per berkas transaksi:")
    st.dataframe(df_detail.head(15), use_container_width=True)

# ------------------------------------------
# PANEL 3: INSIGHT DATA TRANSAKSI
# ------------------------------------------
elif page == "Insight Data Transaksi":
    st.header("📈 Dashboard Analisis Data Finansial Bisnis UMKM")
    st.write("Transformasi data transaksi hasil konversi OCR menjadi grafik intelijen strategis untuk mendukung keputusan manajerial.")

    # Komponen Interaktif Tambahan Nilai Input Margin Laba (BQ2)
    st.markdown("### 💰 Proyeksi Estimasi Nilai Laba Sederhana Dinamis (Insight BQ2)")
    st.write("Pelaku UMKM tidak perlu menginput harga modal beli satu per satu. Masukkan nilai persentase keuntungan kotor di bawah ini untuk melihat simulasi kalkulasi profitabilitas otomatis:")
    
    input_margin = st.number_input("Tentukan Asumsi Persentase Margin Laba Toko (%):", min_value=1, max_value=100, value=20, step=5)
    margin_decimal = input_margin / 100.0

    # Lakukan kalkulasi laba berbasis filter data riil
    if np.issubdtype(df_filtered[col_harga].dtype, np.number):
        total_omzet_real = df_filtered[col_harga].sum()
        total_laba_real = total_omzet_real * margin_decimal
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Akumulasi Omzet Penjualan (Data Terfilter)", f"Rp {total_omzet_real:,.0f}")
        m_col2.metric(f"Proyeksi Estimasi Laba Kotor ({input_margin}%)", f"Rp {total_laba_real:,.0f}")
        m_col3.metric("Rerata Nilai Belanja Per Transaksi", f"Rp {df_filtered[col_harga].mean():,.0f}")
        
        st.markdown(f"""
        > **📋 Studi Kasus Validasi Bisnis (Demo Struk `primer_0079.jpg`):**
        > Berdasarkan data acuan pengujian pada berkas demo struk `primer_0079.jpg` dengan ketetapan margin kotor sebesar **20%**, total nilai omzet belanja kotor terkumpul sebesar **Rp 67.100** dan secara otomatis menghasilkan estimasi keuntungan kotor senilai **Rp 13.420** tanpa pelacakan harga pokok penjualan (HPP) manual.
        """)
    else:
        st.warning("Peringatan: Tipe data pada kolom harga masukan terdeteksi bukan berformat numerik angka kotor, kalkulasi metrik laba global dinonaktifkan.")

    st.markdown("---")
    st.subheader("💡 Jawaban Analisis Kasus Masalah Bisnis Terstruktur (Insight BQ3)")
    
    # Dropdown Kasus Bisnis
    pilihan_analisis = st.selectbox("Pilih Visualisasi Pertanyaan Bisnis:", [
        "Kasus 1: Grafik Distribusi Frekuensi Volume Penjualan Item Produk (Top 10 Terlaris)",
        "Kasus 2: Grafik Distribusi Kelas Kategori Sebaran Harga Produk",
        "Kasus 3: Analisis Proyeksi Akumulasi Laba Bersih per Produk Dagang"
    ])

    if "Kasus 1" in pilihan_analisis:
        st.markdown("#### **Analisis Perputaran Stok Barang Produk (*Product Volume Frequency Distribution*)**")
        
        # Hitung frekuensi produk
        top_produk = df_filtered[col_barang].value_counts().reset_index().head(10)
        top_produk.columns = ['Nama Item Barang', 'Frekuensi Pembelian (Kali)']
        
        fig1 = px.bar(
            top_produk, x='Nama Item Barang', y='Frekuensi Pembelian (Kali)', color='Nama Item Barang',
            title=f"Grafik 2. Top 10 Kuantitas Frekuensi Kemunculan Produk di Struk (Margin Terpilih: {input_margin}%)",
            labels={'Nama Item Barang': "Daftar Identitas Nama Barang", 'Frekuensi Pembelian (Kali)': "Jumlah Kuantitas Kemunculan Terhitung"}
        )
        fig1.update_layout(xaxis_title="Nama Komoditas Produk", yaxis_title="Total Frekuensi Kemunculan (Unit)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **🔍 Keterangan Insight Visualisasi Sumbu & Keputusan Bisnis:**
        * **Sumbu X:** Menunjukkan nama string dari barang dagangan yang teridentifikasi oleh sistem OCR.
        * **Sumbu Y:** Menunjukkan total banyaknya kemunculan item produk tersebut di dalam tumpukan struk belanja.
        * **Pola Data Transaksi:** Data memperlihatkan bahwa mayoritas pola transaksi konsumen bersifat retail satuan (1 unit per baris nota), bukan grosir skala besar. 
        * **Aplikasi Manajerial:** Berdasarkan pola frekuensi ini, item dengan kontribusi volume terbesar seperti makanan ringan kemasan atau minuman dingin instan menunjukkan perputaran arus kas yang cepat (*fast-moving*), sehingga pemilik toko harus mengalokasikan ruang pajang etalase depan dan menjaga ketersediaan stok produk tersebut agar tidak mengalami *out-of-stock*.
        """)

    elif "Kasus 2" in pilihan_analisis:
        st.markdown("#### **Analisis Peta Sebaran Kelas Daya Beli Konsumen (*Price Stratification*)**")
        
        if kolom_filter and kolom_filter in df_filtered.columns:
            # Jika kolom kategori_harga ada di CSV
            dist_kategori = df_filtered[kolom_filter].value_counts().reset_index()
            dist_kategori.columns = ['Kelas Kategori Harga', 'Jumlah Baris Data']
            
            fig2 = px.pie(
                dist_kategori, names='Kelas Kategori Harga', values='Jumlah Baris Data',
                title="Grafik 3. Proporsi Proporsional Pembagian Kelas Kategori Strata Harga Item Produk"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Jika tidak ada kolom kategori_harga, buat visualisasi histogram sebaran harga dari kolom nominal harga
            fig2 = px.histogram(
                df_filtered, x=col_harga, nbins=15, color_discrete_sequence=['#228B22'],
                title="Grafik 3. Distribusi Frekuensi Nilai Rentang Harga Satuan Barang Transaksi",
                labels={col_harga: "Nilai Nominal Harga Jual (Rp)"}
            )
            fig2.update_layout(xaxis_title="Rentang Nominal Nilai Harga Barang (Rp)", yaxis_title="Banyaknya Item Produk Terdeteksi (Baris)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **🔍 Keterangan Insight Visualisasi Sumbu & Keputusan Bisnis:**
        * **Sumbu/Proporsi Visual:** Membagi persebaran variasi harga ke dalam stratifikasi kelas ekonomi ritel.
        * **Pola Sebaran Finansial:** Hasil olahan *cleaning* data menunjukkan bahwa **sekitar 69% item produk berada pada segmen kategori Murah (Rp 5.000 – Rp 20.000) dan kategori Sedang (Rp 20.000 – Rp 50.000)** dengan nilai median harga satuan bertengger di angka **Rp 15.145**. Hanya segelintir komoditas barang yang masuk ke dalam kategori Mahal atau Sangat Mahal (masing-masing hanya mencatatkan 9 item produk).
        * **Aplikasi Manajerial:** Karakteristik ini mencerminkan secara akurat pola belanja kebutuhan primer sehari-hari konsumen di toko kelontong mikro. Pemilik UMKM disarankan untuk memprioritaskan modal kerja mereka pada rentang harga kebutuhan pokok di bawah Rp 50.000 karena di sanalah letak perputaran sirkulasi uang kas utama toko berada.
        """)

    elif "Kasus 3" in pilihan_analisis:
        st.markdown("#### **Analisis Kontribusi Laba Kotor Komoditas Dagang (*Product Profitability Contribution*)**")
        
        if np.issubdtype(df_filtered[col_harga].dtype, np.number):
            # Hitung laba per produk
            df_profit = df_filtered.groupby(col_barang)[col_harga].sum().reset_index()
            df_profit['Estimasi_Laba_Produk'] = df_profit[col_harga] * margin_decimal
            df_profit_sorted = df_profit.sort_values(by='Estimasi_Laba_Produk', ascending=False).head(10)
            
            fig3 = px.bar(
                df_profit_sorted, x=col_barang, y='Estimasi_Laba_Produk', color='Estimasi_Laba_Produk',
                title=f"Grafik 4. Top 10 Produk Penyumbang Nilai Estimasi Laba Terbesar (Asumsi Keuntungan: {input_margin}%)",
                labels={col_barang: "Identitas Nama Komoditas", 'Estimasi_Laba_Produk': "Total Nilai Laba (Rp)"},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig3.update_layout(xaxis_title="Daftar Nama Barang Dagangan", yaxis_title="Total Akumulasi Nilai Keuntungan Bersih (Rp)")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown(f"""
            **🔍 Keterangan Insight Visualisasi Sumbu & Keputusan Bisnis:**
            * **Sumbu X:** Daftar nama barang komoditas dagangan yang berhasil diekstraksi dari struk.
            * **Sumbu Y:** Akumulasi total nilai rupiah proyeksi laba kotor berdasarkan perkalian persentase {input_margin}%.
            * **Pola Perilaku Konsumen:** Berdasarkan data riil pengujian, item dengan kontribusi nilai laba tertinggi didominasi oleh produk pelengkap sekunder yang sering dibeli dalam jumlah lebih dari satu unit per struk transaksi (seperti contoh kasus *Kanzler Bakso Ori* atau *Nutrijell Powder*). Pola bisnis ini mengindikasikan bahwa volume pembelian kuantitas per baris struk memberikan kontribusi profitabilitas kumulatif yang jauh lebih signifikan bagi toko dibandingkan barang berharga satuan tinggi namun jarang dibeli.
            * **Aplikasi Manajerial:** Pendekatan sistem cerdas satu angka masukan persentase margin laba ini memberikan kepraktisan luar biasa bagi pemilik UMKM yang tidak memiliki pencatatan pembukuan harga modal beli yang rapi—cukup dengan memasukkan satu angka target keuntungan toko, sistem visualisasi ini langsung menyuguhkan laporan laba rugi instan untuk pembukuan pembagian keuntungan bulanan yang sehat.
            """)
        else:
            st.error("Gagal memproses visualisasi profitabilitas kotor. Kolom nilai harga jual di dalam berkas CSV primer kamu terdeteksi mengandung format teks alfabet (*non-numeric*). Harap lakukan pembersihan ulang.")

    st.markdown("---")
    st.caption("⚠️ **Catatan Evaluasi Lapangan Pengawas Akhir:** Beberapa entitas pembacaan string nama identitas toko dan nilai angka total transaksi final pada berkas primer dilaporkan masih menyisakan komponen kesalahan karakter kecil (*noise OCR residual*). Untuk pengembangan sistem jangka panjang berikutnya, disarankan menerapkan pembersihan tingkat lanjut menggunakan teknik pencocokan berbasis kedekatan string kata (*fuzzy matching algorithm*).")

# Footer Hak Cipta Sidang Laporan
st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Finansial NOPI v1.0 • Fakultas Teknologi Industri • Teknik Informatika Universitas Gunadarma © 2026")

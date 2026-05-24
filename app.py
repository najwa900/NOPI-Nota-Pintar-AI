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
    # Identifikasi kolom barang dan harga secara dinamis
    col_barang = 'nama_barang' if 'nama_barang' in df_primer.columns else df_primer.columns[0]
    col_harga = 'total_harga' if 'total_harga' in df_primer.columns else df_primer.columns[-1]
    
    # Deteksi kolom kategori harga
    kolom_filter = 'kategori_harga' if 'kategori_harga' in df_primer.columns else None

    # Filter Global di Sidebar (Hanya aktif untuk Panel 3)
    if page == "Insight Data Transaksi":
        st.sidebar.subheader("🎛️ Filter Interaktif Bisnis")
        
        # 1. Filter Kategori / Nama Item
        if kolom_filter:
            opsi_kategori = df_primer[kolom_filter].dropna().unique().tolist()
            selected_cat = st.sidebar.multiselect("Pilih Kategori Harga:", opsi_kategori, default=opsi_kategori)
            df_filtered = df_primer[df_primer[kolom_filter].isin(selected_cat)]
        else:
            opsi_barang = df_primer[col_barang].dropna().unique().tolist()[:15]
            selected_items = st.sidebar.multiselect("Pilih Sampel Produk:", opsi_barang, default=opsi_barang[:5])
            df_filtered = df_primer[df_primer[col_barang].isin(selected_items)]
        
        # 2. Filter Rentang Tanggal (PENGAMAN DARI TYPEERROR)
        if 'Tanggal_Clean' in df_primer.columns and not df_primer['Tanggal_Clean'].isna().all():
            min_date = df_primer['Tanggal_Clean'].min().date()
            max_date = df_primer['Tanggal_Clean'].max().date()
            
            date_range = st.sidebar.date_input("Rentang Tanggal Transaksi:", [min_date, max_date])
            
            # PENGAMAN: Filter hanya dieksekusi jika kedua rentang tanggal (Mulai & Selesai) sudah dipilih
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date = date_range[0]
                end_date = date_range[1]
                
                # Gunakan .dt.date agar perbandingan tipe data sebanding antara Pandas dan Streamlit
                df_filtered = df_filtered[
                    (df_filtered['Tanggal_Clean'].dt.date >= start_date) & 
                    (df_filtered['Tanggal_Clean'].dt.date <= end_date)
                ]
else:
    st.error(f"Gagal memuat berkas dari folder `data/`. Deskripsi Error: {error_msg}")
    st.stop()

# ==========================================
# 3. KONTEN DASHBOARD UTAMA
# ==========================================
st.title("🚀 Dashboard Intelijen Bisnis - NOPI (Nota Pintar)")
st.markdown("**Solusi Otomasi Finansial UMKM Berbasis Integrasi Klasifikasi Citra CNN & Ekstraksi PaddleOCR**")
st.markdown("---")

# ------------------------------------------
# PANEL 1: ANALISIS PIPELINE SISTEM
# ------------------------------------------
if page == "Analisis Pipeline Sistem":
    st.header("⚙️ Kerangka Kerja Pipeline Sistem Dua Tahap")
    
    with st.expander("📖 Latar Belakang & Pernyataan Masalah Proyek (Klik untuk Membaca)", expanded=True):
        st.markdown("""
        * **Konteks Makro:** Indonesia memiliki 65,5 juta UMKM yang menyumbang 61% terhadap PDB nasional. Namun, sebagian besar pelaku usaha mikro/toko kelontong masih mengandalkan ingatan atau rekap manual kertas fisik.
        * **Permasalahan Utama:** Proses rekap finansial lambat, rentan kesalahan manusia (*human error*), serta tidak tersedianya visualisasi profit secara *real-time*.
        * **Solusi Arsitektur NOPI:** Membangun *pipeline* otomatis dua tahap:
          1. **Jaringan CNN Classifier:** Bertindak sebagai *gatekeeper* untuk memfilter secara otomatis gambar input pengguna dan memisahkan gambar struk belanja asli dari gambar non-struk (*noise*).
          2. **Ekstraksi Mesin OCR (PaddleOCR):** Mengambil teks tidak terstruktur dari gambar struk yang lolos seleksi dan mengubahnya menjadi informasi data tabel terstruktur (Nama Item, Harga Satuan, Total Transaksi).
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🛡️ Tahap 1: Klasifikasi Citra (CNN)")
        st.info("""
        * **Fungsi Utama:** Melakukan filtrasi di awal sistem sebelum gambar diproses oleh OCR.
        * **Target Performa:** Memisahkan dokumen struk dan non-struk dengan target akurasi minimal 85%.
        * **Dampak Bisnis:** Menolak gambar sampah di awal rantai komputasi sehingga meringankan beban pemrosesan server cloud.
        """)
        
    with col2:
        st.subheader("📝 Tahap 2: Ekstraksi Informasi Struktural (OCR)")
        st.info("""
        * **Fungsi Utama:** Mengenali teks, memetakan koordinat biner karakter angka, serta melakukan parsing otomatis.
        * **Output Struktural:** Mengonversi string nota menjadi data siap guna yang mencakup nama item produk, kuantitas beli, harga satuan, dan total harga.
        * **Dampak Bisnis:** Automasi pembukuan tanpa entri manual, menghemat waktu rekap harian secara signifikan.
        """)

    st.markdown("---")
    st.subheader("📋 Contoh Basis Data Terstruktur Hasil Ekstraksi Akhir (`Dataset_Terstruktur_Primer_NOPI.csv`)")
    st.write("Berikut adalah 10 baris sampel data transaksi riil yang berhasil dikonversi oleh sistem pipeline:")
    st.dataframe(df_primer.head(10), use_container_width=True)
    
    st.markdown("""
    **💡 Narasi Analisis Alur Kerja:**
    Integrasi taktik hibrida sekuensial ini menjamin efisiensi pengolahan data transaksi. Pembagian kerja yang tegas memastikan model OCR hanya menghabiskan daya komputasinya untuk memproses citra yang telah divalidasi sebagai dokumen transaksi berharga bisnis oleh modul CNN.
    """)

# ------------------------------------------
# PANEL 2: PERFORMA MODEL AI
# ------------------------------------------
elif page == "Performa Model AI":
    st.header("📊 Metrik Evaluasi Performa Model Pengenalan Karakter")
    st.write("Analisis perbandingan performa model berdasarkan pengujian riil berkas benchmark.")

    # Indikator Pengukuran Business Question 1 (BQ1)
    st.subheader("📌 Indikator Pengukuran Business Question 1 (BQ1)")
    st.markdown("""
    Pengukuran efektivitas model OCR dilakukan secara komparatif mengacu pada 4 indikator utama: *success rate* parsing, akurasi baris item, akurasi kalkulasi total harga, dan durasi kecepatan waktu proses (*inference time*).
    """)

    # Visualisasi Utama Komparasi Model OCR
    col_x = df_eval.columns[0]
    col_y = df_eval.columns[1]
    
    fig_ev = px.bar(
        df_eval, x=col_x, y=col_y, color=col_x,
        title="Grafik 1. Perbandingan Tingkat Keberhasilan Pemrosesan Model OCR (%)",
        labels={col_x: "Arsitektur Model OCR", col_y: "Persentase Keberhasilan (%)"}
    )
    fig_ev.update_layout(xaxis_title="Nama Arsitektur Model AI", yaxis_title="Success Rate (%)")
    st.plotly_chart(fig_ev, use_container_width=True)

    # Narasi Jawaban Eksplisit BQ1
    st.markdown("""
    #### **📖 Interpretasi Laporan Performa Model (Insight BQ1):**
    Berdasarkan hasil evaluasi komparasi tiga arsitektur model, ditemukan jawaban eksplisit untuk **BQ1**:
    * **Model Terpilih (PaddleOCR):** Memiliki performa yang **paling seimbang** untuk arsitektur NOPI. PaddleOCR memperoleh *success rate* tertinggi sebesar **73.33%** dan akurasi total nominal harga tertinggi sebesar **26.09%**, meskipun waktu pemrosesan komputasinya sedikit lebih lama daripada Tesseract.
    * **Kelemahan Tesseract:** Meskipun mencatat waktu proses paling cepat (**2.64 detik**), Tesseract menghasilkan tingkat akurasi jumlah item dan total harga **paling rendah**, menjadikannya kurang andal untuk pelaporan keuangan yang sensitif terhadap kesalahan nominal angka.
    * **Kelemahan EasyOCR:** Mampu menghasilkan akurasi jumlah item tertinggi (**33.5%**), namun waktu pemrosesan komputasinya **sangat lambat (18.73 detik)** serta akurasi total harganya sangat rendah (**14.0%**), sehingga tidak efisien untuk operasional harian (*production deployment*).
    
    * **Kesimpulan Finansial:** **PaddleOCR** dipilih sebagai model terbaik karena menjamin akurasi pembacaan angka transaksi kotor yang lebih stabil dengan toleransi waktu tunggu pemrosesan yang masuk akal bagi pengguna.
    """)
    
    st.markdown("---")
    st.subheader("📋 Tabel Parameter Komparasi Model Akhir (`hasil_komparasi_ocr_final.csv`)")
    st.dataframe(df_komparasi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Rincian Log Akurasi Per Berkas Uji (`detail_akurasi_semua_model.csv`)")
    st.write("Berikut adalah 15 baris data log hasil pelacakan performa per berkas struk:")
    st.dataframe(df_detail.head(15), use_container_width=True)

# ------------------------------------------
# PANEL 3: INSIGHT DATA TRANSAKSI
# ------------------------------------------
elif page == "Insight Data Transaksi":
    st.header("📈 Analisis Finansial & Dashboard Insight Transaksi")
    st.write("Transformasi data mentah hasil pembacaan OCR menjadi informasi intelijen taktis pembukuan UMKM.")

    # Komponen Masukan Persentase Laba (BQ2)
    st.markdown("### 💰 Proyeksi Estimasi Nilai Laba Sederhana Dinamis (Insight BQ2)")
    st.write("Pelaku usaha dapat langsung memproyeksikan laba kotor toko tanpa perlu menginput harga modal beli satu per satu. Tentukan target margin keuntungan di bawah ini:")
    
    input_margin = st.number_input("Input Target Margin Laba Toko (%):", min_value=1, max_value=100, value=20, step=5)
    margin_decimal = input_margin / 100.0

    # Kalkulasi Metrik Global Berdasarkan Data Riil Terfilter
    if np.issubdtype(df_filtered[col_harga].dtype, np.number):
        total_omzet_real = df_filtered[col_harga].sum()
        total_laba_real = total_omzet_real * margin_decimal
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Akumulasi Omzet Penjualan", f"Rp {total_omzet_real:,.0f}")
        m_col2.metric(f"Estimasi Laba Kotor Ritel ({input_margin}%)", f"Rp {total_laba_real:,.0f}")
        m_col3.metric("Rerata Nilai Belanja Konsumen per Item", f"Rp {df_filtered[col_harga].mean():,.0f}")
        
        # Jawaban Eksplisit BQ2
        st.markdown(f"""
        #### **📖 Interpretasi Perhitungan Laba Praktis (Insight BQ2):**
        * **Jawaban Eksplisit BQ2:** Melalui pendekatan pemodelan satu angka persentase margin laba dari pengguna, sistem dapat langsung mengestimasi laba per item produk secara otomatis. 
        * **Studi Kasus Pembuktian Riil:** Berdasarkan pengujian acuan pada demo berkas struk `primer_0079.jpg` dengan ketetapan margin **20%**, total nilai omzet kotor struk sebesar **Rp 67.100** berhasil mencetak proyeksi laba bersih sebesar **Rp 13.420**.
        * **Pola Profitabilitas:** Data riil menunjukkan kontribusi keuntungan terbesar disumbang oleh produk komoditas yang dibeli konsumen dalam jumlah lebih dari 1 unit per transaksi (misal: *Kanzler Bakso Ori* atau *Nutrijell Powder*). Pola ini membuktikan bahwa volume kuantitas transaksi retail harian berbobot jauh lebih besar terhadap akumulasi profitabilitas toko kelontong dibandingkan barang berharga mahal yang perputarannya lambat.
        """)
    else:
        st.warning("Kolom harga terdeteksi bukan berformat angka numerik penuh, kalkulasi visualisasi otomatis dinonaktifkan.")

    st.markdown("---")
    st.subheader("💡 Jawaban Analisis Kasus Masalah Bisnis Terstruktur (Insight BQ3)")
    
    pilihan_analisis = st.selectbox("Pilih Visualisasi Grafik Pertanyaan Bisnis:", [
        "Kasus 1: Grafik Distribusi Frekuensi Volume Penjualan Item Produk (Top 10 Terlaris)",
        "Kasus 2: Grafik Distribusi Kelas Kategori Sebaran Harga Produk",
        "Kasus 3: Analisis Proyeksi Akumulasi Laba Bersih per Produk Dagang"
    ])

    if "Kasus 1" in pilihan_analisis:
        st.markdown("#### **Analisis Frekuensi Perputaran Produk (*Product Volume Frequency Distribution*)**")
        
        top_produk = df_filtered[col_barang].value_counts().reset_index().head(10)
        top_produk.columns = ['Nama Item Barang', 'Frekuensi Pembelian']
        
        fig1 = px.bar(
            top_produk, x='Nama Item Barang', y='Frekuensi Pembelian', color='Nama Item Barang',
            title=f"Grafik 2. Top 10 Kuantitas Frekuensi Kemunculan Produk di Struk (Margin Terpilih: {input_margin}%)",
            labels={'Nama Item Barang': "Daftar Identitas Nama Komoditas Barang", 'Frekuensi Pembelian': "Total Frekuensi Kemunculan (Kali)"}
        )
        fig1.update_layout(xaxis_title="Nama Komoditas Item Produk", yaxis_title="Total Kuantitas Baris Kemunculan")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **🔍 Keterangan Kaitan Sumbu Grafik & Jawaban Eksplisit BQ3 (Bagian Tren Volume):**
        * **Sumbu X (Horizontal):** Memetakan string nama unik dari barang-barang dagangan hasil pembacaan OCR.
        * **Sumbu Y (Vertikal):** Menunjukkan jumlah akumulasi baris kemunculan barang tersebut di dalam database transaksi.
        * **Jawaban Eksplisit Kasus 1:** Hasil analisis pembacaan memperlihatkan bahwa mayoritas data transaksi bersifat retail eceran satuan (1 unit barang per baris nota), bukan grosir skala besar. 
        * **Aplikasi Manajerial:** Melalui grafik tren volume ini, pemilik usaha mikro dapat langsung memetakan barang yang perputarannya cepat (*fast-moving product*). Keputusan strategisnya adalah mengoptimalkan penataan etalase depan toko serta menjaga kontinuitas stok komoditas tersebut agar terhindar dari potensi kehilangan momentum penjualan.
        """)

    elif "Kasus 2" in pilihan_analisis:
        st.markdown("#### **Analisis Segmentasi Strata Harga Barang (*Price Stratification*)**")
        
        if kolom_filter and kolom_filter in df_filtered.columns:
            dist_kategori = df_filtered[kolom_filter].value_counts().reset_index()
            dist_kategori.columns = ['Kelas Kategori Harga', 'Jumlah Data']
            
            fig2 = px.pie(
                dist_kategori, names='Kelas Kategori Harga', values='Jumlah Data',
                title="Grafik 3. Proporsi Pembagian Kelas Kategori Strata Harga Item Produk"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.histogram(
                df_filtered, x=col_harga, nbins=15, color_discrete_sequence=['#2E8B57'],
                title="Grafik 3. Distribusi Frekuensi Nilai Rentang Harga Satuan Barang Transaksi",
                labels={col_harga: "Nilai Nominal Harga Jual Produk (Rp)"}
            )
            fig2.update_layout(xaxis_title="Rentang Nominal Harga Barang (Rp)", yaxis_title="Banyaknya Item Produk Terdeteksi (Baris)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **🔍 Keterangan Kaitan Sumbu Grafik & Jawaban Eksplisit BQ3 (Bagian Distribusi Strata):**
        * **Proporsi Visual / Sumbu:** Membagi sebaran nominal harga barang retail ke dalam klaster kelas ekonomi.
        * **Jawaban Eksplisit Kasus 2:** Hasil rekapitulasi data riil menunjukkan **sekitar 69% item produk bertumpu kuat pada kategori segmen Murah (Rp 5.000 – Rp 20.000) dan kategori Sedang (Rp 20.000 – Rp 50.000)** dengan perolehan nilai median harga satuan di angka **Rp 15.145**. Variasi komoditas berstatus Mahal atau Sangat Mahal hanya mencatatkan kuantitas minoritas (masing-masing hanya 9 item barang).
        * **Aplikasi Manajerial:** Pola ini membuktikan secara empiris bahwa basis konsumen toko kelontong didominasi oleh pemenuhan belanja komoditas kebutuhan pokok berskala kecil. Pemilik UMKM sebaiknya memfokuskan alokasi perputaran modal kerja harian pada pengadaan barang di rentang harga di bawah Rp 50.000 karena di sanalah pusat sirkulasi transaksi kas aktif berada.
        """)

    elif "Kasus 3" in pilihan_analisis:
        st.markdown("#### **Analisis Kontribusi Keuntungan Produk (*Product Profitability Contribution*)**")
        
        if np.issubdtype(df_filtered[col_harga].dtype, np.number):
            df_profit = df_filtered.groupby(col_barang)[col_harga].sum().reset_index()
            df_profit['Estimasi_Laba_Produk'] = df_profit[col_harga] * margin_decimal
            df_profit_sorted = df_profit.sort_values(by='Estimasi_Laba_Produk', ascending=False).head(10)
            
            fig3 = px.bar(
                df_profit_sorted, x=col_barang, y='Estimasi_Laba_Produk', color='Estimasi_Laba_Produk',
                title=f"Grafik 4. Top 10 Produk Penyumbang Nilai Estimasi Laba Terbesar (Asumsi Keuntungan: {input_margin}%)",
                labels={col_barang: "Identitas Nama Komoditas", 'Estimasi_Laba_Produk': "Total Nilai Laba Kotor (Rp)"},
                color_continuous_scale=px.colors.sequential.YlGnBu
            )
            fig3.update_layout(xaxis_title="Daftar Nama Barang Dagangan", yaxis_title="Total Akumulasi Keuntungan Bersih (Rp)")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown(f"""
            **🔍 Keterangan Kaitan Sumbu Grafik & Jawaban Eksplisit BQ3 (Bagian Profitabilitas):**
            * **Sumbu X (Horizontal):** Representasi string deretan nama barang dagangan retail hasil ekstraksi.
            * **Sumbu Y (Vertikal):** Akumulasi kalkulasi nominal nilai laba rupiah hasil kalkulasi formula persentase {input_margin}%.
            * **Jawaban Eksplisit Kasus 3:** Data mengonfirmasi bahwa visualisasi pengolahan data transaksi ini berhasil diagregasi secara utuh menjadi laporan ringkas per struk belanja terstruktur yang memuat nama toko, tanggal, kuantitas item, akumulasi total omzet transaksi, dan estimasi keuntungannya secara otomatis.
            * **Aplikasi Manajerial:** Sistem visualisasi interaktif ini memecahkan kebuntuan administrasi pelaku UMKM yang tidak memiliki sistem basis data HPP atau harga modal terstruktur. Cukup menetapkan satu parameter margin laba global, pemilik usaha langsung disuguhkan laporan performa laba per produk komoditas yang siap dijadikan landasan evaluasi performa bisnis pembukuan bulanan secara mandiri.
            """)
        else:
            st.error("Proses pembuatan grafik profitabilitas gagal. Format data kolom harga di CSV kamu dilaporkan mengandung noise alfabet (*non-numeric*).")

    st.markdown("---")
    st.caption("⚠️ **Catatan Evaluasi Lapangan Pengawas Akhir:** Beberapa entitas pembacaan string nama identitas toko dan nilai angka total transaksi final pada berkas primer dilaporkan masih menyisakan komponen kesalahan karakter kecil (*noise OCR residual*). Untuk pengembangan sistem jangka panjang berikutnya, disarankan menerapkan pembersihan tingkat lanjut menggunakan teknik pencocokan berbasis kedekatan string kata (*fuzzy matching algorithm*).")

# Footer Hak Cipta Sidang Laporan
st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Finansial NOPI v1.0 • Fakultas Teknologi Industri • Teknik Informatika Universitas Gunadarma © 2026")

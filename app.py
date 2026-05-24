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
    # Identifikasi nama kolom secara aman dan toleran terhadap variasi huruf besar/kecil
    col_barang = None
    for c in df_primer.columns:
        if 'barang' in c.lower() or 'item' in c.lower() or 'nama' in c.lower():
            col_barang = c
            break
    if not col_barang: col_barang = df_primer.columns[0]

    col_harga = None
    for c in df_primer.columns:
        if 'harga' in c.lower() or 'total' in c.lower() or 'amount' in c.lower():
            col_harga = c
            break
    if not col_harga: col_harga = df_primer.columns[-1]
    
    # Memastikan kolom harga bertipe numerik (membersihkan noise string/titik/koma jika ada)
    if df_primer[col_harga].dtype == 'object':
        df_primer[col_harga] = df_primer[col_harga].astype(str).str.replace(r'[^\d]', '', regex=True)
        df_primer[col_harga] = pd.to_numeric(df_primer[col_harga], errors='coerce').fillna(0)

    kolom_filter = None
    for c in df_primer.columns:
        if 'kategori' in c.lower() or 'cluster' in c.lower():
            kolom_filter = c
            break

    # Filter Global di Sidebar (Hanya aktif untuk Panel 3)
    if page == "Insight Data Transaksi":
        st.sidebar.subheader("🎛️ Filter Interaktif Bisnis")
        df_filtered = df_primer.copy()
        
        # 1. Filter Kategori / Nama Item
        if kolom_filter:
            opsi_kategori = df_primer[kolom_filter].dropna().unique().tolist()
            selected_cat = st.sidebar.multiselect("Pilih Kategori Harga:", opsi_kategori, default=opsi_kategori)
            df_filtered = df_filtered[df_filtered[kolom_filter].isin(selected_cat)]
        else:
            opsi_barang = df_primer[col_barang].dropna().unique().tolist()[:15]
            selected_items = st.sidebar.multiselect("Pilih Sampel Produk:", opsi_barang, default=opsi_barang[:5])
            df_filtered = df_filtered[df_filtered[col_barang].isin(selected_items)]
        
        # 2. Filter Rentang Tanggal yang Aman
        if 'Tanggal_Clean' in df_primer.columns and not df_primer['Tanggal_Clean'].isna().all():
            min_date = df_primer['Tanggal_Clean'].min().date()
            max_date = df_primer['Tanggal_Clean'].max().date()
            
            date_range = st.sidebar.date_input("Rentang Tanggal Transaksi:", [min_date, max_date])
            
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                df_filtered = df_filtered[
                    (df_filtered['Tanggal_Clean'].dt.date >= date_range[0]) & 
                    (df_filtered['Tanggal_Clean'].dt.date <= date_range[1])
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
    
    with st.expander("📖 Latar Belakang & Pernyataan Masalah Proyek", expanded=True):
        st.markdown("""
        * **Konteks Makro:** Indonesia memiliki 65,5 juta UMKM yang menyumbang 61% terhadap PDB nasional. Namun, sebagian besar pelaku usaha mikro masih mengandalkan ingatan atau rekap manual kertas fisik.
        * **Permasalahan Utama:** Proses rekap finansial lambat, rentan kesalahan manusia (*human error*), serta tidak tersedianya visualisasi profit secara *real-time*.
        * **Solusi Arsitektur NOPI:** Membangun *pipeline* otomatis dua tahap:
          1. **Jaringan CNN Classifier:** Memfilter secara otomatis gambar input pengguna untuk memisahkan gambar struk belanja dari gambar non-struk (*noise*).
          2. **Ekstraksi Mesin OCR (PaddleOCR):** Mengubah teks tidak terstruktur dari gambar struk menjadi informasi data tabel terstruktur.
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🛡️ Tahap 1: Klasifikasi Citra (CNN)")
        st.info("Memisahkan dokumen struk dan non-struk dengan target akurasi minimal 85% untuk meringankan beban pemrosesan server.")
    with col2:
        st.subheader("📝 Tahap 2: Ekstraksi Informasi Struktural (OCR)")
        st.info("Mengonversi gambar nota menjadi data tabel siap guna yang mencakup nama item produk, harga, dan total belanja.")

    st.markdown("---")
    st.subheader("📋 Contoh Basis Data Terstruktur Hasil Ekstraksi Akhir (`Dataset_Terstruktur_Primer_NOPI.csv`)")
    st.dataframe(df_primer.head(10), use_container_width=True)

# ------------------------------------------
# PANEL 2: PERFORMA MODEL AI
# ------------------------------------------
# ------------------------------------------
# PANEL 2: PERFORMA MODEL AI (Pembuktian BQ1, BQ2, BQ3)
# ------------------------------------------
elif page == "Performa Model AI":
    st.header("📊 Metrik Evaluasi Performa Model Pengenalan Karakter")
    st.write("Laporan evaluasi komparatif komprehensif sebagai bukti ilmiah penunjang jawaban seluruh Business Question (BQ).")
    st.markdown("---")

    # ============================================================
    # PEMBUKTIAN BQ1: PERBANDINGAN MODEL OCR
    # ============================================================
    st.subheader("📌 1. Pembuktian BQ1: Perbandingan & Akurasi Model OCR")
    
    # Deteksi Kolom File evaluasi_3_model.csv atau hasil_komparasi_ocr_final.csv secara aman
    # Kita asumsikan menggunakan data dari df_eval untuk visualisasi grafik utama Success Rate
    col_x = df_eval.columns[0]
    col_y = df_eval.columns[1]
    
    col_metric1, col_metric2 = st.columns([2, 1])
    
    with col_metric1:
        fig_ev = px.bar(
            df_eval, x=col_x, y=col_y, color=col_x,
            title="Grafik 1. Perbandingan Tingkat Keberhasilan Pemrosesan Model OCR (%)",
            labels={col_x: "Arsitektur Model OCR", col_y: "Persentase Keberhasilan (%)"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_ev.update_layout(xaxis_title="Nama Arsitektur Model AI", yaxis_title="Success Rate (%)")
        st.plotly_chart(fig_ev, use_container_width=True)
    
    with col_metric2:
        st.markdown("**Metrik Kunci Pilihan (PaddleOCR):**")
        st.metric(label="Success Rate Tertinggi", value="73.33%", delta="Unggul vs Lainnya")
        st.metric(label="Akurasi Total Harga", value="26.09%", delta="Tertinggi")
        st.metric(label="Waktu Proses (Inference)", value="~ Sedang", delta="Slower vs Tesseract", delta_color="inverse")

    st.markdown("""
    #### **📖 Interpretasi Laporan Performa Model (Jawaban Eksplisit BQ1):**
    Berdasarkan data grafik benchmark di atas, teknologi OCR dimanfaatkan melalui seleksi arsitektur terbaik untuk memastikan tingkat akurasi transaksi:
    * **Model Terpilih (PaddleOCR):** Menunjukkan performa **paling seimbang**. PaddleOCR memperoleh *success rate* tertinggi sebesar **73.33%** dan akurasi total nominal harga tertinggi sebesar **26.09%**. Karakteristik ini sangat krusial bagi UMKM karena meminimalkan kesalahan fatal pada pencatatan nominal uang.
    * **Tesseract OCR:** Meskipun mencatat durasi waktu pemrosesan paling cepat (**2.64 detik**), model ini menghasilkan tingkat akurasi jumlah baris item dan total harga **paling rendah**, sehingga kurang andal untuk ekstraksi data finansial terstruktur.
    * **EasyOCR:** Berhasil memperoleh akurasi jumlah item tertinggi (**33.5%**), namun waktu pemrosesan komputasinya **sangat lambat (18.73 detik)** dan akurasi total harganya rendah (**14.0%**), sehingga tidak efisien untuk kebutuhan operasional harian (*production deployment*).
    """)
    
    st.markdown("---")

    # ============================================================
    # PEMBUKTIAN BQ2: FORMULASI ESTIMASI LABA SEBAGAI SOLUSI BISNIS
    # ============================================================
    st.subheader("📌 2. Pembuktian BQ2: Validasi Pendekatan Estimasi Laba")
    st.markdown("""
    Pelaku usaha mikro memerlukan cara mengetahui laba secara efisien tanpa sistem pencatatan Harga Pokok Penjualan (HPP) / harga beli yang rumit. 
    NOPI membuktikannya melalui integrasi formula rekayasa fitur (*feature engineering*) langsung menggunakan asumsi persentase margin seragam pada data terstruktur hasil ekstraksi.
    """)
    
    # Menampilkan visualisasi ringkas benchmark total harga vs potensi laba dari file komparasi jika ada datanya
    # Jika tidak, kita gunakan penjelasan pembuktian empiris berbasis file hasil_komparasi_ocr_final.csv
    st.markdown("""
    #### **📖 Bukti Empiris Formulasi Finansial (Jawaban Eksplisit BQ2):**
    Sistem mengotomatisasi kalkulasi laba melalui *pipeline* kode di mana variabel `total_harga_item` hasil OCR langsung dikonversikan menjadi produk informasi keuangan baru.
    * **Bukti Studi Kasus Struk `primer_0079.jpg`:** Berdasarkan baris data pengujian pada file komparasi, penetapan parameter input tunggal berupa **margin kotor 20%** terbukti sukses mengkalkulasi total omzet struk kotor sebesar **Rp 67.100** menjadi estimasi profit bersih senilai **Rp 13.420** secara instan.
    * **Signifikansi Praktis:** UMKM ritel tidak perlu melakukan pencatatan inventori stok barang masuk satu per satu—cukup masukkan satu angka margin rata-rata toko, dan visualisasi distribusi laba per produk langsung tersaji otomatis untuk laporan keuangan pembagian hasil bulanan.
    """)
    
    st.markdown("---")

    # ============================================================
    # PEMBUKTIAN BQ3: TRANSFORMASI DATA MENJADI LAPORAN TERSTRUKTUR
    # ============================================================
    st.subheader("📌 3. Pembuktian BQ3: Struktur Tabel Finansial & Data Bersih")
    st.markdown("""
    Bagaimana data hasil OCR dapat diolah menjadi laporan keuangan terstruktur? Buktinya terdapat pada proses pembersihan (*data wrangling*) 
    dan agregasi baris data transaksi yang tersimpan di dalam file komparasi dan dataset primer di bawah ini.
    """)
    
    # Menampilkan tabel hasil_komparasi_ocr_final.csv sebagai bukti fisik database terstruktur
    st.markdown("**Tabel Tabel Parameter Benchmark & Komparasi Kualitas Ekstraksi Akhir (`hasil_komparasi_ocr_final.csv`)**")
    st.dataframe(df_komparasi, use_container_width=True)
    
    st.markdown("""
    #### **📖 Interpretasi Data Hasil Wrangling (Jawaban Eksplisit BQ3):**
    Data transaksi mentah dari PaddleOCR berhasil ditransformasikan menjadi laporan terstruktur bernilai bisnis tinggi setelah melewati beberapa tahap pembersihan:
    1. **Segmentasi Kelas Ekonomi (Strata Harga):** Rekayasa fitur membagi produk secara otomatis. Bukti visual pada data transaksi memperlihatkan **69% data menumpuk pada kategori Murah (Rp 5rb–20rb) dan Sedang (Rp 20rb–50rb)** dengan nilai median harga satuan bertengger di angka **Rp 15.145**.
    2. **Identifikasi Perilaku Belanja:** Database membuktikan mayoritas pola transaksi konsumen bersifat retail eceran satuan (1 unit per baris nota), bukan grosir skala besar.
    3. **Konsolidasi Multi-Toko:** Sistem berhasil merekapitulasi keberagaman entitas toko dari struk belanja (mulai dari minimarket, warung kelontong, kafe, hingga apotek) menjadi satu format laporan keuangan seragam yang memuat: *Nama Toko, Tanggal Valid, Jumlah Jenis Item, Total Nilai Transaksi, dan Estimasi Profit.*
    
    > **⚠️ Catatan Teknis untuk Sidang:** Proses *data wrangling* ini sukses menyaring *noise* pembacaan ekstrem (seperti baris dengan kuantitas > 200 unit akibat salah baca OCR telah dihapus secara otomatis). Beberapa residu karakter kecil pada nama toko disarankan ditangani menggunakan algoritma *fuzzy matching* pada riset kelanjutan.
    """)
    
    st.markdown("---")
    st.subheader("📋 Log Tambahan Rincian Akurasi Berkas Uji (`detail_akurasi_semua_model.csv`)")
    st.write("Tabel di bawah ini memuat bukti log rincian tingkat akurasi karakter per berkas struk sebagai validasi data dukung utama:")
    st.dataframe(df_detail.head(15), use_container_width=True)

# ------------------------------------------
# PANEL 3: INSIGHT DATA TRANSAKSI
# ------------------------------------------
elif page == "Insight Data Transaksi":
    st.header("📈 Analisis Finansial & Dashboard Insight Transaksi")
    
    # BAGIAN PROYEKSI LABA (BQ2)
    st.markdown("### 💰 Proyeksi Estimasi Nilai Laba Sederhana Dinamis (Insight BQ2)")
    input_margin = st.number_input("Input Target Margin Laba Toko (%):", min_value=1, max_value=100, value=20, step=5)
    margin_decimal = input_margin / 100.0

    # Kalkulasi Metrik Finansial Riil
    total_omzet_real = df_filtered[col_harga].sum()
    total_laba_real = total_omzet_real * margin_decimal
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Akumulasi Omzet Penjualan", f"Rp {total_omzet_real:,.0f}")
    m_col2.metric(f"Estimasi Laba Kotor Ritel ({input_margin}%)", f"Rp {total_laba_real:,.0f}")
    m_col3.metric("Rerata Nilai Belanja Konsumen per Item", f"Rp {df_filtered[col_harga].mean():,.0f}")
    
    st.markdown("""
    #### **📖 Interpretasi Perhitungan Laba Praktis (Insight BQ2):**
    * **Jawaban Eksplisit BQ2:** Melalui pendekatan satu angka persentase margin laba, sistem langsung menghitung estimasi laba per item secara efisien tanpa pelacakan HPP manual.
    * **Studi Kasus Riil:** Pada demo struk `primer_0079.jpg` dengan margin **20%**, total omzet kotor sebesar **Rp 67.100** otomatis menghasilkan estimasi laba kotor senilai **Rp 13.420**.
    """)

    st.markdown("---")
    st.subheader("💡 Bukti Distribusi Data Transaksi Terstruktur (Insight BQ3)")
    st.markdown("Untuk menjawab **BQ3**, berikut ditampilkan seluruh grafik distribusi dan tren data finansial secara bersamaan:")

    # MEMUNCULKAN SEMUA GRAFIK DISTRIBUSI BQ3 SEKALIGUS (TIDAK BERGANTIAN)
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### **1. Distribusi Kuantitas Frekuensi Produk (Top 10)**")
        top_produk = df_filtered[col_barang].value_counts().reset_index().head(10)
        top_produk.columns = [col_barang, 'Frekuensi Pembelian']
        
        fig1 = px.bar(
            top_produk, x=col_barang, y='Frekuensi Pembelian', color=col_barang,
            title="Grafik 2. Top 10 Kuantitas Frekuensi Kemunculan Produk di Struk",
            labels={col_barang: "Daftar Identitas Nama Barang", 'Frekuensi Pembelian': "Total Frekuensi"}
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("* **Insight Volume:** Pola transaksi bersifat retail eceran satuan (1 unit per baris nota). Barang *fast-moving* harus selalu dijaga ketersediaannya di etalase toko.")

    with g_col2:
        st.markdown("#### **2. Analisis Segmentasi Strata Harga Barang**")
        if kolom_filter and kolom_filter in df_filtered.columns:
            dist_kategori = df_filtered[kolom_filter].value_counts().reset_index()
            dist_kategori.columns = [kolom_filter, 'Jumlah Data']
            fig2 = px.pie(dist_kategori, names=kolom_filter, values='Jumlah Data', title="Grafik 3. Proporsi Kelas Kategori Strata Harga")
        else:
            fig2 = px.histogram(df_filtered, x=col_harga, nbins=15, title="Grafik 3. Distribusi Frekuensi Rentang Harga Satuan")
            fig2.update_layout(xaxis_title="Rentang Harga (Rp)", yaxis_title="Banyaknya Item")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("* **Insight Strata:** **Sekitar 69% item produk berada pada segmen Murah (5rb–20rb) & Sedang (20rb–50rb)** dengan median **Rp 15.145**, mencerminkan daya beli harian untuk kebutuhan pokok harian.")

    st.markdown("---")
    st.markdown("#### **3. Analisis Kontribusi Keuntungan Bersih per Komoditas Dagang**")
    
    df_profit = df_filtered.groupby(col_barang)[col_harga].sum().reset_index()
    df_profit['Estimasi_Laba_Produk'] = df_profit[col_harga] * margin_decimal
    df_profit_sorted = df_profit.sort_values(by='Estimasi_Laba_Produk', ascending=False).head(15)
    
    fig3 = px.bar(
        df_profit_sorted, x=col_barang, y='Estimasi_Laba_Produk', color='Estimasi_Laba_Produk',
        title=f"Grafik 4. Top 15 Produk Penyumbang Nilai Estimasi Laba Terbesar (Margin: {input_margin}%)",
        color_continuous_scale=px.colors.sequential.YlGnBu
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Insight Profitabilitas:** Agregasi data berhasil menyajikan performa laba rugi instan per komoditas siap guna untuk pembukuan UMKM secara mandiri.")

    st.markdown("---")
    st.caption("⚠️ **Catatan Evaluasi:** Beberapa nama toko masih menyisakan komponen kesalahan karakter kecil (*noise OCR residual*). Direkomendasikan menggunakan teknik *fuzzy matching algorithm* untuk tahap pengembangan selanjutnya.")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Finansial NOPI v1.0 • Universitas Gunadarma © 2026")

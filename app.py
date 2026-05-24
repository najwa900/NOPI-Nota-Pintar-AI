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
    
    # Pengondisian kolom tanggal jika ada
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
# 2. SIDEBAR NAVIGASI & FILTER INTERAKTIF
# ==========================================
st.sidebar.title("🔍 Menu Utama NOPI")
page = st.sidebar.radio("Pilih Panel Dashboard:", 
    ["Analisis Pipeline Sistem", "Performa Model AI", "Insight Data Transaksi"])

st.sidebar.markdown("---")

# Filter Interaktif (Hanya aktif di Panel 3 / Insight Data)
if page == "Insight Data Transaksi" and data_loaded:
    st.sidebar.subheader("🎛️ Filter Interaktif")
    
    kolom_filter = 'kategori_harga' if 'kategori_harga' in df_primer.columns else df_primer.columns[1]
    opsi_kategori = df_primer[kolom_filter].dropna().unique().tolist()
    selected_cat = st.sidebar.multiselect("Pilih Kategori/Item:", opsi_kategori, default=opsi_kategori[:3])
    
    if 'Tanggal_Clean' in df_primer.columns and not df_primer['Tanggal_Clean'].isna().all():
        min_date = df_primer['Tanggal_Clean'].min().date()
        max_date = df_primer['Tanggal_Clean'].max().date()
        date_range = st.sidebar.date_input("Rentang Tanggal Trx:", [min_date, max_date])
    else:
        st.sidebar.info("Kolom tanggal otomatis menggunakan indeks default.")

# ==========================================
# 3. KONTEN UTAMA DASHBOARD
# ==========================================
st.title("🚀 NOPI: Nota Pintar Dashboard")
st.markdown("Integrasi Arsitektur CNN (Gambar) & PaddleOCR (Teks) untuk Otomasi UMKM")
st.markdown("---")

if not data_loaded:
    st.error(f"Gagal memuat file dari folder `data/`. Pastikan penulisan file di GitHub sudah benar. Error: {error_msg}")
    st.stop()

# ------------------------------------------
# PANEL 1: ANALISIS PIPELINE SISTEM
# ------------------------------------------
if page == "Analisis Pipeline Sistem":
    st.header("⚙️ Alur Kerja Arsitektur Pipeline Dua Tahap")
    st.write("Penjelasan struktural bagaimana sistem NOPI memproses citra nota belanja secara sekuensial.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Tahap 1: Klasifikasi Citra (CNN)")
        st.markdown("""
        Arsitektur **CNN (Convolutional Neural Network)** diimplementasikan di awal sistem sebagai *gatekeeper*.
        - **Fungsi Utama:** Melakukan klasifikasi biner otomatis untuk mendeteksi tipe dokumen input.
        - **Sistem Filter:** Memisahkan gambar struk legal dari citra *non-struk* (seperti foto pemandangan, wajah, atau dokumen non-transaksi).
        - **Tujuan:** Mencegah terjadinya *noise* data atau kegagalan sistem pemrosesan karakter teks sebelum masuk ke tahap berikutnya.
        """)
        
    with col2:
        st.subheader("Tahap 2: Ekstraksi Teks Struktural (OCR)")
        st.markdown("""
        Citra yang telah divalidasi sebagai struk resmi oleh CNN kemudian diteruskan ke mesin **PaddleOCR**.
        - **Fungsi Utama:** Mengenali karakter teks pada gambar dan memetakan koordinat biner secara akurat.
        - **Struktur Output:** Teks mentah hasil pembacaan dikonversi menjadi data tabel terstruktur yang mencakup komponen penting:
          1. Nama Item / Produk
          2. Harga Satuan
          3. Total Transaksi Belanja
        """)
    
    st.markdown("---")
    st.subheader("📄 Contoh Struktur Data Terstruktur Hasil Ekstraksi Akhir (`Dataset_Terstruktur_Primer_NOPI.csv`)")
    st.dataframe(df_primer.head(10), use_container_width=True)
            
    st.info("**Narasi Kesimpulan Panel 1:** Pipeline dua tahap ini menjamin efisiensi beban kerja komputasi. Jaringan CNN menyaring dan memastikan validitas dokumen secara cepat, sehingga model OCR hanya memproses gambar berisikan teks transaksi yang bernilai bisnis.")

# ------------------------------------------
# PANEL 2: PERFORMA MODEL AI
# ------------------------------------------
elif page == "Performa Model AI":
    st.header("📊 Integrasi Metrik & Evaluasi Model AI")
    st.write("Laporan performa pengujian riil berdasarkan berkas benchmark model.")
    
    st.subheader("1. Komparasi Arsitektur OCR")
    st.write("Visualisasi berdasarkan data akurasi pada file `evaluasi_3_model.csv`:")
    
    x_col = df_eval.columns[0]
    y_col = df_eval.columns[1]
    
    fig_ev = px.bar(df_eval, x=x_col, y=y_col, color=x_col, 
                    title="Tingkat Akurasi / Keberhasilan Ekstraksi Karakter")
    st.plotly_chart(fig_ev, use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. Detail Benchmark Riil Berdasarkan `hasil_komparasi_ocr_final.csv`")
    st.dataframe(df_komparasi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("3. Rincian Data Akurasi Per File (`detail_akurasi_semua_model.csv`)")
    st.dataframe(df_detail.head(15), use_container_width=True)

    st.info("**Narasi Kesimpulan Panel 2:** Berdasarkan metrik performa eksperimen, **PaddleOCR** menunjukkan hasil akurasi dan stabilitas ekstraksi karakter terbaik pada kondisi teks struk yang buram atau memiliki pencahayaan minim jika dibandingkan dengan model Tesseract dan EasyOCR.")

# ------------------------------------------
# PANEL 3: INSIGHT DATA TRANSAKSI
# ------------------------------------------
elif page == "Insight Data Transaksi":
    st.header("📈 Dashboard Insight Transaksi Bisnis UMKM")
    st.write("Analisis keputusan bisnis berbasis statistik data transaksi yang berhasil diekstrak oleh sistem NOPI.")
    
    df_filtered = df_primer.copy()
    if selected_cat:
        df_filtered = df_filtered[df_filtered[kolom_filter].isin(selected_cat)]
        
    st.subheader("💡 Pertanyaan & Jawaban Kasus Bisnis")
    pertanyaan = st.selectbox("Pilih Analisis Masalah Bisnis:", [
        "1. Produk apa saja yang memiliki frekuensi kemunculan tertinggi di struk?",
        "2. Bagaimana peta sebaran nilai transaksi belanja konsumen?",
        "3. Berapa estimasi profitabilitas per item barang (Asumsi Margin Laba)?"
    ])
    
    col_barang = 'nama_barang' if 'nama_barang' in df_filtered.columns else df_filtered.columns[0]
    col_harga = 'total_harga' if 'total_harga' in df_filtered.columns else df_filtered.columns[-1]

    if "1." in pertanyaan:
        st.markdown("**Jawaban Analisis:** Tren volume memperlihatkan komoditas utama yang paling sering dibeli oleh pelanggan.")
        top_items = df_filtered[col_barang].value_counts().reset_index().head(10)
        top_items.columns = ['Nama Item', 'Jumlah Kemunculan']
        
        fig1 = px.bar(top_items, x='Nama Item', y='Jumlah Kemunculan', color='Nama Item', title="Top 10 Item Terlaris")
        st.plotly_chart(fig1, use_container_width=True)
        
    elif "2." in pertanyaan:
        st.markdown("**Jawaban Analisis:** Distribusi harga menunjukkan kelas daya beli konsumen terhadap barang yang dijual.")
        fig2 = px.histogram(df_filtered, x=col_harga, nbins=20, title="Distribusi Nilai Transaksi Belanja")
        st.plotly_chart(fig2, use_container_width=True)
        
    elif "3." in pertanyaan:
        st.markdown("**Jawaban Analisis:** Proyeksi margin keuntungan kotor rata-rata sebesar 15% dari total penjualan per produk.")
        if np.issubdtype(df_filtered[col_harga].dtype, np.number):
            df_filtered['Estimasi_Laba'] = df_filtered[col_harga] * 0.15
            fig3 = px.bar(df_filtered.head(15), x=col_barang, y='Estimasi_Laba', title="Estimasi Margin Keuntungan per Produk (Top 15)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Kolom harga terdeteksi bukan angka numerik. Harap periksa format data.")

    st.markdown("""
    ***
    **Narasi Kesimpulan Panel 3:**
    Melalui filter interaktif kategori, pemilik UMKM dapat langsung memetakan barang yang perputarannya cepat (*fast-moving product*) guna mengoptimalkan manajemen stok gudang agar terhindar dari kerugian akibat barang kedaluwarsa.
    """)

# Footer Aplikasi
st.sidebar.caption("Sistem Aplikasi NOPI v1.0 • Universitas Gunadarma © 2026")

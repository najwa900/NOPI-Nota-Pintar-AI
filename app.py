import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set Page Config
st.set_page_config(page_title="NOPI - Nota Pintar Dashboard", layout="wide")

# --- FUNGSI HELPER UNTUK CLEANING (Sesuai Notebook) ---
def clean_nama_barang(val):
    if pd.isna(val): return None
    val = str(val).strip()
    val = re.sub(r'^[^a-zA-Z0-9]+', '', val)
    val = re.sub(r'[^a-zA-Z0-9\s\-\.&()/]+', ' ', val)
    return val.strip() if len(val) >= 3 else None

def kategorikan_harga(h):
    if h <= 5000: return 'Sangat Murah (<=5rb)'
    elif h <= 20000: return 'Murah (5-20rb)'
    elif h <= 50000: return 'Sedang (20-50rb)'
    elif h <= 100000: return 'Mahal (50-100rb)'
    else: return 'Sangat Mahal (>100rb)'

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df_primer = pd.read_csv('Dataset_Terstruktur_Primer_NOPI.csv')
    df_evaluasi = pd.read_csv('evaluasi_3_model.csv')
    df_detail = pd.read_csv('detail_akurasi_semua_model.csv')
    
    # Pre-cleaning sederhana untuk dashboard
    df_clean = df_primer.dropna(subset=['nama_barang', 'harga_satuan']).copy()
    df_clean['nama_barang'] = df_clean['nama_barang'].apply(clean_nama_barang)
    df_clean = df_clean[df_clean['harga_satuan'] >= 500]
    df_clean['kategori_harga'] = df_clean['harga_satuan'].apply(kategorikan_harga)
    
    return df_primer, df_evaluasi, df_detail, df_clean

try:
    df_primer, df_evaluasi, df_detail, df_clean = load_data()
except Exception as e:
    st.error(f"Gagal memuat file CSV. Pastikan file tersedia. Error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🚀 NOPI Dashboard")
st.sidebar.info("Aplikasi berbasis OCR untuk membantu manajemen keuangan UMKM.")
menu = st.sidebar.selectbox("Navigasi", ["Home", "BQ1: Performa OCR", "BQ2: Estimasi Laba", "BQ3: Laporan Transaksi"])

# --- TAB HOME ---
if menu == "Home":
    st.title("📊 Dashboard Analisis Data: NOPI (Nota Pintar)")
    st.markdown("""
    Pencatatan keuangan manual sering menyebabkan *human error* bagi pelaku UMKM. 
    **NOPI** hadir sebagai solusi berbasis AI (CNN & OCR) untuk otomatisasi ekstraksi data struk belanja.
    
    **Tujuan Dashboard:**
    1. Membuktikan akurasi ekstraksi OCR.
    2. Mendemonstrasikan transparansi perhitungan laba.
    3. Menyajikan data transaksi yang terstruktur.
    """)
    st.image("https://via.placeholder.com/800x200.png?text=Nota+Pintar+UMKM+Digital", use_column_width=True)

# --- BQ1: PERFORMA OCR ---
elif menu == "BQ1: Performa OCR":
    st.header("🔍 BQ1: Bagaimana performa teknologi OCR dalam mengekstrak informasi?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Perbandingan Model")
        st.dataframe(df_evaluasi)
    
    with col2:
        st.write("**Insight:** PaddleOCR dipilih karena memiliki *Success Rate* tertinggi (73.3%) dibandingkan Tesseract dan EasyOCR.")

    st.divider()
    
    # Grafik Bar Metrik Performa
    st.subheader("Visualisasi Metrik Akurasi")
    metrik = st.selectbox("Pilih Metrik", ["Success Rate (%)", "Rata-rata Waktu (Detik)", "Akurasi Total Harga (%)"])
    
    fig, ax = plt.subplots()
    sns.barplot(x="Nama Model", y=metrik, data=df_evaluasi, palette="viridis", ax=ax)
    st.pyplot(fig)

# --- BQ2: ESTIMASI LABA ---
elif menu == "BQ2: Estimasi Laba":
    st.header("💰 BQ2: Bagaimana UMKM mengetahui estimasi laba secara efisien?")
    
    st.write("Sistem menghitung laba berdasarkan asumsi margin yang diinput pengguna.")
    
    margin = st.slider("Input Asumsi Margin Laba (%)", 5, 50, 20)
    
    # Hitung Laba
    df_laba = df_clean[['nama_barang', 'jumlah_barang', 'harga_satuan', 'total_harga_item']].copy()
    df_laba['Laba Estimasi'] = df_laba['total_harga_item'] * (margin/100)
    
    st.subheader(f"Estimasi Laba per Item (Margin {margin}%)")
    st.dataframe(df_laba.head(10))
    
    # Grafik Laba Teratas
    top_laba = df_laba.sort_values('Laba Estimasi', ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x="Laba Estimasi", y="nama_barang", data=top_laba, palette="magma", ax=ax2)
    ax2.set_title("Top 10 Produk Berdasarkan Kontribusi Laba")
    st.pyplot(fig2)

# --- BQ3: LAPORAN TRANSAKSI ---
elif menu == "BQ3: Laporan Transaksi":
    st.header("📋 BQ3: Bagaimana data OCR diolah menjadi laporan terstruktur?")
    
    # Metrik Ringkasan
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Item Bersih", len(df_clean))
    m2.metric("Median Harga", f"Rp {df_clean['harga_satuan'].median():,.0f}")
    m3.metric("Item Paling Banyak Dibeli", df_clean['nama_barang'].mode()[0])
    
    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Kategori Harga")
        fig3, ax3 = plt.subplots()
        df_clean['kategori_harga'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax3, colors=sns.color_palette("pastel"))
        ax3.set_ylabel("")
        st.pyplot(fig3)
        
    with col_b:
        st.subheader("Data Transaksi Terstruktur (Final)")
        st.write("Data ini siap digunakan untuk pembukuan digital.")
        st.dataframe(df_clean[['nama_toko', 'tanggal', 'nama_barang', 'harga_satuan', 'kategori_harga']].head(15))

# Footer
st.caption("Copyright © 2024 | Proyek NOPI AI Analysis Dashboard")

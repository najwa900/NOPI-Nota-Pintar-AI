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
    df_primer = pd.read_csv('data/Dataset_Terstruktur_Primer_NOPI.csv')
    df_evaluasi = pd.read_csv('data/evaluasi_3_model.csv')
    df_detail = pd.read_csv('data/detail_akurasi_semua_model.csv')
    
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
# --- BQ1: PERFORMA OCR ---
elif menu == "BQ1: Performa OCR":
    st.header("🔍 BQ1: Bagaimana performa teknologi OCR dalam mengekstrak informasi?")
    
    # Menampilkan Tabel Utama Komparasi
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Tabel Komparasi Performa Model OCR")
        st.dataframe(df_evaluasi, use_container_width=True)
    
    with col2:
        st.subheader("Rekomendasi Utama")
        st.success("""
        **PaddleOCR** dipilih sebagai model produksi karena menawarkan kombinasi metrik yang paling stabil, tingkat keberhasilan parsing data komoditas tertinggi, serta waktu komputasi yang masih dapat ditoleransi.
        """)

    st.divider()
    
    # 1. VISUALISASI GRID BAR CHART (2x2)
    st.subheader("📊 Komparasi Performa 3 Model OCR (Grid Metrics)")
    
    models = df_evaluasi['Nama Model']
    colors_bar = ['steelblue', 'tomato', 'mediumseagreen']

    metrics_list = [
        ('Success Rate (%)', 'Success Rate (%)', 'Persentase (%)', '%.1f%%'),
        ('Rata-rata Waktu (Detik)', 'Rata-rata Waktu Proses (Detik)', 'Detik', '%.2f'),
        ('Akurasi Jumlah Item (%)', 'Akurasi Jumlah Item (%)', 'Persentase (%)', '%.1f%%'),
        ('Akurasi Total Harga (%)', 'Akurasi Total Harga (%)', 'Persentase (%)', '%.1f%%')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (col, title, ylabel, fmt) in zip(axes, metrics_list):
        bars = ax.bar(
            models,
            df_evaluasi[col],
            color=colors_bar,
            alpha=0.85,
            edgecolor='white'
        )
        ax.bar_label(bars, fmt=fmt, padding=3)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel)
        if '%' in col:
            ax.set_ylim(0, 100)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # 2. VISUALISASI PIE CHART DISTRIBUSI STATUS PARSING
    st.subheader("🍕 Distribusi Status Parsing per Model OCR")
    
    # Ambil data pengelompokkan status berdasarkan df_detail
    status_counts = df_detail.groupby(['Model', 'Status']).size().unstack(fill_value=0)
    
    label_map = {
        'Sebagian (Terekstrak tapi ada miss)': 'Sebagian',
        'Sempurna (100%)': 'Sempurna',
        'Gagal Total (0%)': 'Gagal Total'
    }
    status_counts = status_counts.rename(columns=label_map)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    colors_status = ['steelblue', 'mediumseagreen', 'tomato']
    model_names = ['Paddle', 'Tesseract', 'EasyOCR']

    for ax, model in zip(axes2, model_names):
        if model in status_counts.index:
            data = status_counts.loc[model]
            # Sinkronisasi warna jika ada kategori status yang kosong
            current_colors = colors_status[:len(data)]
            
            ax.pie(
                data.values,
                labels=data.index,
                autopct='%1.1f%%',
                colors=current_colors,
                startangle=90,
                textprops={'fontsize': 10}
            )
            ax.set_title(f'Status Parsing — {model}', fontweight='bold', fontsize=12)
        else:
            ax.text(0.5, 0.5, f'Data {model}\nTidak Ditemukan', ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # 3. BAGIAN INSIGHT RESMI BQ1
    st.subheader("💡 Insight Analisis Pertanyaan Bisnis 1")
    st.markdown("""
    Berdasarkan hasil evaluasi pembuktian di atas, **Paddle memiliki performa paling seimbang** dibandingkan Tesseract dan EasyOCR. 
    * **Paddle** memperoleh *success rate* tertinggi sebesar **73.33%** dan akurasi total harga tertinggi sebesar **26.09%**, meskipun waktu prosesnya sedikit lebih lama dibandingkan Tesseract.
    * **Tesseract** memiliki waktu proses paling cepat (**2.64 detik**), tetapi akurasi jumlah item dan total harga paling rendah sehingga kurang andal untuk ekstraksi data transaksi nyata.
    * **EasyOCR** memiliki akurasi jumlah item tertinggi (**33.5%**), namun waktu prosesnya sangat lambat (**18.73 detik**) dan akurasi total harga paling rendah (**14.0%**), sehingga tidak efisien untuk kebutuhan *deployment* sistem.

    **Kesimpulan Dokumen:** Dengan demikian, **Paddle** resmi dipilih sebagai arsitektur OCR untuk proyek **NOPI (Nota Pintar)** ini karena memiliki titik temu keseimbangan terbaik (*trade-off*) antara keberhasilan parsing, akurasi nilai harga finansial, dan efisiensi waktu pemrosesan yang wajar.
    """)

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

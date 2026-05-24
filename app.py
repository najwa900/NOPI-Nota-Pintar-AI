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
    st.markdown("Sistem menghitung perkiraan laba secara otomatis berdasarkan total pengeluaran item struk belanja dan persentase margin keuntungan standar.")
    
    # 1. REPLIKASI LOGIKA DEMO COLAB (Mengunci Data Hanya pada Struk primer_0079.jpg)
    sample_file = 'primer_0079.jpg'
    
    # Ambil data murni dari df_primer agar urutan indeksnya asli seperti di Colab
    sample_struk = df_primer[df_primer['filename'] == sample_file].copy()
    
    # Margin dikunci pada 20% (persen_margin=0.20) sesuai demo ujian di notebook
    margin_tetap = 20
    sample_struk['laba_total'] = sample_struk['total_harga_item'] * (margin_tetap / 100)
    
    # VISUALISASI ESTIMASI LABA PER ITEM (Top 15 Horizontal Bar Chart)
    st.subheader(f"📊 Top 15 Komoditas Berdasarkan Estimasi Laba (Demo Struk: {sample_file})")
    
    # Logika filter, pengurutan, dan tail(15) disamakan 100% dengan notebook
    df_plot = (
        sample_struk[sample_struk['laba_total'] > 0]
        .sort_values('laba_total')
        .tail(15)
    ).copy()
    
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Casting string secara inline untuk mencegah TypeError pada matplotlib Streamlit web
        bars = ax.barh(
            [str(x) for x in df_plot['nama_barang']], 
            df_plot['laba_total'].tolist(),
            color='mediumseagreen',
            alpha=0.85,
            edgecolor='white'
        )
        
        # Atribut visualisasi grafis disamakan 100% dengan Colab
        ax.bar_label(bars, fmt='Rp %.0f', padding=3, fontsize=9)
        ax.set_title('BQ2 — Estimasi Laba per Item (Margin 20%)', fontweight='bold')
        ax.set_xlabel('Estimasi Laba Total (Rp)')
        
        plt.tight_layout()
        st.pyplot(fig) # Render langsung ke kanvas web Streamlit
    else:
        st.warning(f"Tidak ada data transaksi valid pada file {sample_file} untuk ditampilkan.")

    st.divider()

    # ### Insight BQ2 RESMI DARI NOTEBOOK ###
    st.subheader("💡 Insight Analisis Pertanyaan Bisnis 2")
    st.markdown("""
    Dengan input persentase margin dari pengguna, sistem dapat langsung menghitung estimasi laba per item tanpa perlu memasukkan harga beli satu per satu. Berdasarkan demo struk `primer_0079.jpg` dengan margin 20%, total omzet struk sebesar **Rp 67.100** menghasilkan estimasi laba **Rp 13.420**.

    Item dengan kontribusi laba tertinggi adalah **Kanzlr Bakso Ori 48G** (Rp 3.480) karena dibeli 2 unit, diikuti **Nutrijel Pwd.Strw.15** (Rp 2.640). Pola ini menunjukkan bahwa item dengan jumlah beli lebih dari 1 unit berkontribusi lebih besar terhadap total laba meskipun harga satuannya tidak selalu tertinggi.

    Pendekatan ini praktis untuk pelaku UMKM yang tidak memiliki sistem pencatatan harga beli terstruktur — cukup input satu angka margin, sistem langsung menghasilkan laporan laba per item yang siap digunakan untuk pembukuan sederhana.
    """)

# --- BQ3: LAPORAN TRANSAKSI ---
elif menu == "BQ3: Laporan Transaksi":
    st.header("📋 BQ3: Bagaimana data OCR diolah menjadi laporan terstruktur untuk mendukung pengambilan keputusan bisnis??")
    
    # Perlu import matplotlib ticker untuk memformat sumbu X pada histogram
    import matplotlib.ticker as mticker

    # 1. VISUALISASI GRID: DISTRIBUSI DATA TRANSAKSI
    df_harga = df_clean[df_clean['harga_satuan'] > 0]
    median_harga = df_harga['harga_satuan'].median()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Distribusi harga satuan
    axes[0].hist(
        df_harga['harga_satuan'],
        bins=30,
        color='steelblue',
        alpha=0.8,
        edgecolor='white'
    )

    axes[0].axvline(
        median_harga,
        color='tomato',
        linestyle='--',
        linewidth=2,
        label=f'Median: Rp {median_harga:,.0f}'
    )

    axes[0].set_title('Distribusi Harga Satuan Item', fontweight='bold')
    axes[0].set_xlabel('Harga Satuan (Rp)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}rb')
    )
    axes[0].legend()

    # Chart 2: Distribusi jumlah barang
    jumlah_counts = df_clean['jumlah_barang'].value_counts().sort_index().head(10)

    bars = axes[1].bar(
        jumlah_counts.index.astype(str),
        jumlah_counts.values,
        color='mediumseagreen',
        alpha=0.85,
        edgecolor='white'
    )

    axes[1].bar_label(bars, padding=3)
    axes[1].set_title('Distribusi Jumlah Barang per Baris Transaksi', fontweight='bold')
    axes[1].set_xlabel('Jumlah Barang')
    axes[1].set_ylabel('Frekuensi')

    plt.suptitle('BQ3 — Distribusi Data Transaksi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig) # Render langsung ke kanvas Streamlit

    st.divider()

    # 2. VISUALISASI SEGMENTASI KATEGORI HARGA
    kat_order = [
        'Sangat Murah (<=5rb)',
        'Murah (5-20rb)',
        'Sedang (20-50rb)',
        'Mahal (50-100rb)',
        'Sangat Mahal (>100rb)'
    ]

    kat_counts = df_clean['kategori_harga'].value_counts().reindex(kat_order, fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    colors_kat = [
        'steelblue',
        'mediumseagreen',
        'orange',
        'tomato',
        'mediumpurple'
    ]

    bars2 = ax2.barh(
        kat_counts.index,
        kat_counts.values,
        color=colors_kat,
        alpha=0.85,
        edgecolor='white'
    )

    ax2.bar_label(bars2, padding=3, fontsize=10)
    ax2.set_title('BQ3 — Segmentasi Item berdasarkan Kategori Harga', fontweight='bold')
    ax2.set_xlabel('Jumlah Item')

    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # 3. VISUALISASI TOP STRUK BERDASARKAN TOTAL TRANSAKSI
    # Memastikan dataframe laporan_struk_filtered terbuat murni dari basis kolom dataset asli
    group_col = 'filename' if 'filename' in df_clean.columns else 'nama_toko'
    
    laporan_struk_filtered = df_clean.groupby(group_col).agg(
        total_transaksi=('total_harga_item', 'sum')
    ).reset_index()
    
    if group_col != 'filename':
        laporan_struk_filtered = laporan_struk_filtered.rename(columns={group_col: 'filename'})

    top_struk = (
        laporan_struk_filtered
        .sort_values('total_transaksi', ascending=False)
        .head(10)
        .sort_values('total_transaksi')
    )

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    bars3 = ax3.barh(
        top_struk['filename'],
        top_struk['total_transaksi'],
        color='mediumpurple',
        alpha=0.85,
        edgecolor='white'
    )

    ax3.bar_label(bars3, fmt='Rp %.0f', padding=3, fontsize=9)
    ax3.set_title('BQ3 — Struk Teratas berdasarkan Total Transaksi Valid', fontweight='bold')
    ax3.set_xlabel('Total Transaksi (Rp)')

    plt.tight_layout()
    st.pyplot(fig3)

    st.divider()

    # 4. INSIGHT BQ3 RESMI DARI NOTEBOOK
    st.subheader("💡 Insight Analisis Pertanyaan Bisnis 3")
    st.markdown("""
    Data transaksi hasil OCR berhasil diolah menjadi laporan terstruktur setelah melalui proses *cleaning* dan *feature engineering*.

    Sekitar 69% item berada pada kategori Murah (5–20rb) dan Sedang (20–50rb) dengan median harga satuan Rp 15.145, mencerminkan pola belanja kebutuhan sehari-hari. Hanya sebagian kecil item masuk kategori Mahal dan Sangat Mahal, masing-masing 9 item.

    Mayoritas transaksi bersifat satuan (1 unit per baris), bukan grosir. Dataset mencakup berbagai jenis toko mulai dari minimarket, warung, kafe, hingga apotek.

    Data dapat diagregasi menjadi laporan ringkas per struk yang memuat nama toko, tanggal, total item, total transaksi, dan estimasi laba. Dengan asumsi margin 20%, sistem dapat langsung menghasilkan estimasi laba tanpa input harga beli manual, sehingga praktis untuk pembukuan sederhana pelaku UMKM.

    > **Catatan:** Beberapa nama toko dan nilai total transaksi masih mengandung *noise* OCR residual. Normalisasi nama toko lebih lanjut dapat dilakukan menggunakan teknik *fuzzy matching* pada tahap pengembangan berikutnya.
    """)

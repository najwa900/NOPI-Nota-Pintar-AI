import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # path file
    path_primer = os.path.join(DATA_DIR, "Dataset_Terstruktur_Primer_NOPI.csv")
    path_evaluasi = os.path.join(DATA_DIR, "evaluasi_3_model.csv")
    path_detail = os.path.join(DATA_DIR, "detail_akurasi_semua_model.csv")
    path_clean = os.path.join(DATA_DIR, "dataset_ocr_clean_final.csv")
    path_all = os.path.join(DATA_DIR, "all_images_metadata.csv")

    # cek file wajib
    required_files = [
        path_primer,
        path_evaluasi,
        path_detail,
        path_clean,
        path_all
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File tidak ditemukan: {file}")

    # load csv
    df_primer = pd.read_csv(path_primer)
    df_evaluasi = pd.read_csv(path_evaluasi)
    df_detail = pd.read_csv(path_detail)

    # dataset utama
    df_clean = pd.read_csv(
        path_clean,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    # pastikan kategori_harga ada
    if 'kategori_harga' not in df_clean.columns:
        df_clean['kategori_harga'] = df_clean['harga_satuan'].apply(kategorikan_harga)

    # metadata gambar
    df_all = pd.read_csv(path_all)

    # kategori harga metadata
    if 'harga_satuan' in df_all.columns:
        df_all['kategori_harga'] = df_all['harga_satuan'].apply(kategorikan_harga)

    return (
        df_primer,
        df_evaluasi,
        df_detail,
        df_clean,
        df_all
    )


# --- LOAD SEMUA DATA ---
try:
    (
        df_primer,
        df_evaluasi,
        df_detail,
        df_clean,
        df_all
    ) = load_data()

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()
# --- SIDEBAR NAVIGASI ---
st.sidebar.title("🚀 NOPI Dashboard")
st.sidebar.info("Aplikasi berbasis OCR untuk membantu manajemen keuangan UMKM.")
menu = st.sidebar.selectbox(
    "Navigasi Halaman:", 
    [
        "Home", 
        "Ringkasan & EDA", 
        "BQ1: Performa OCR", 
        "BQ2: Estimasi Laba", 
        "BQ3: Laporan Transaksi"
    ]
)

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

# --- 1. HALAMAN RINGKASAN & EDA ---
elif menu == "Ringkasan & EDA":
    st.title("📊 EDA — Komposisi Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Distribusi Kelas (Struk vs Non-Struk)")
        fig1, ax1 = plt.subplots(figsize=(7, 5))

        label_counts = df_all['label'].value_counts()

        ax1.bar(
            label_counts.index,
            label_counts.values,
            color=['#4CAF50', '#FF5722'],
            edgecolor='white'
        )

        for i, v in enumerate(label_counts.values):
            ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')

        st.pyplot(fig1)

    with col2:
        st.write("### Distribusi Sumber Data (Source)")
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        src_counts = df_all['source'].value_counts()
        colors = ['#2196F3', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4']

        ax2.bar(
            src_counts.index,
            src_counts.values,
            color=colors[:len(src_counts)],
            edgecolor='white'
        )

        st.pyplot(fig2)

    st.info(
        "**Insight:** Dataset memiliki keseimbangan kelas yang sempurna (50:50), "
        "yang sangat baik untuk menghindari bias pada model klasifikasi."
    )

# --- BQ1: PERFORMA OCR ---
elif menu == "BQ1: Performa OCR":
    st.header("🔍 BQ1: Bagaimana performa teknologi OCR dalam mengekstrak informasi?")

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

    st.subheader("🍕 Distribusi Status Parsing per Model OCR")

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
    st.markdown("Sistem menampilkan perkiraan laba berdasarkan data terstruktur hasil ekstraksi OCR yang telah melalui proses *cleaning* akhir.")

    sample_file = 'primer_0079.jpg'
    sample_struk = df_clean[df_clean['filename'] == sample_file].copy()

    if 'laba_total' not in sample_struk.columns:
        if 'estimasi_laba_20' in sample_struk.columns:
            sample_struk['laba_total'] = sample_struk['estimasi_laba_20']
        else:
            sample_struk['laba_total'] = sample_struk['total_harga_item'] * 0.20

    st.subheader(f"📊 Demo Hasil Perhitungan Laba (Struk: {sample_file})")

    kolom_tampilan = ['nama_toko', 'tanggal', 'nama_barang', 'jumlah_barang', 'harga_satuan', 'total_harga_item', 'laba_total']
    kolom_tersedia = [col for col in kolom_tampilan if col in sample_struk.columns]

    st.write("Tabel representasi data terstruktur transaksional item pada demo struk (Dataset Bersih Final):")
    st.dataframe(sample_struk[kolom_tersedia], use_container_width=True)

    st.divider()

    st.write("### Visualisasi Top 15 Komoditas Berdasarkan Estimasi Laba")

    df_plot = (
        sample_struk[sample_struk['laba_total'] > 0]
        .sort_values('laba_total')
        .tail(15)
    ).copy()

    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.barh(
            [str(x) for x in df_plot['nama_barang']], 
            df_plot['laba_total'].tolist(),
            color='mediumseagreen',
            alpha=0.85,
            edgecolor='white'
        )

        ax.bar_label(bars, fmt='Rp %.0f', padding=3, fontsize=9)
        ax.set_title('BQ2 — Estimasi Laba per Item (Margin 20%)', fontweight='bold')
        ax.set_xlabel('Estimasi Laba Total (Rp)')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning(f"Tidak ada data transaksi valid pada file {sample_file} untuk divisualisasikan.")

    st.divider()

    st.subheader("💡 Insight Analisis Pertanyaan Bisnis 2")
    st.markdown("""
    Dengan input persentase margin dari pengguna, sistem dapat langsung menghitung estimasi laba per item tanpa perlu memasukkan harga beli satu per satu. Berdasarkan demo struk `primer_0079.jpg` dengan margin 20%, total omzet struk sebesar **Rp 67.100** menghasilkan estimasi laba **Rp 13.420**.

    Item dengan kontribusi laba tertinggi adalah **Kanzlr Bakso Ori 48G** (Rp 3.480) karena dibeli 2 unit, diikuti **Nutrijel Pwd.Strw.15** (Rp 2.640). Pola ini menunjukkan bahwa item dengan jumlah beli lebih dari 1 unit berkontribusi lebih besar terhadap total laba meskipun harga satuannya tidak selalu tertinggi.

    Pendekatan ini praktis untuk pelaku UMKM yang tidak memiliki sistem pencatatan harga beli terstruktur — cukup input satu angka margin, sistem langsung menghasilkan laporan laba per item yang siap digunakan untuk pembukuan sederhana.
    """)

# --- BQ3: LAPORAN TRANSAKSI ---
elif menu == "BQ3: Laporan Transaksi":
    st.header("📋 BQ3: Bagaimana data OCR diolah menjadi laporan terstruktur untuk mendukung pengambilan keputusan bisnis?")
    st.markdown("Mentransformasikan hasil ekstraksi teks acak dokumen nota belanja menjadi laporan finansial terstruktur untuk menunjang strategi bisnis UMKM.")
    
    import matplotlib.ticker as mticker

    # ==========================================
    # LOGIKA AGREGASI DATA (SINKRONISASI MURNI DARI DATASET CLEAR)
    # ==========================================
    # Menghitung estimasi laba jika belum ada kolomnya di dataset_ocr_clear_final
    if 'estimasi_laba_20' not in df_clean.columns:
        df_clean['estimasi_laba_20'] = df_clean['total_harga_item'] * 0.20

    # Ambil pengelompokan agregasi per berkas nota/struk langsung dari data bersih
    laporan_struk = df_clean.groupby('filename').agg(
        nama_toko=('nama_toko', 'first') if 'nama_toko' in df_clean.columns else ('filename', 'first'),
        tanggal_clean=('tanggal_clean', 'first') if 'tanggal_clean' in df_clean.columns else (('tanggal', 'first') if 'tanggal' in df_clean.columns else ('filename', 'first')),
        tanggal_valid=('tanggal_valid', 'first') if 'tanggal_valid' in df_clean.columns else ('filename', 'count'),
        jumlah_item=('nama_barang', 'count'),
        total_qty=('jumlah_barang', 'sum') if 'jumlah_barang' in df_clean.columns else ('nama_barang', 'count'),
        total_transaksi=('total_harga_item', 'sum'),
        estimasi_laba=('estimasi_laba_20', 'sum')
    ).reset_index()

    # Kriteria filter pembatasan data agar terbebas dari text noise hasil OCR
    cond_date = (laporan_struk['tanggal_valid'] == True) if 'tanggal_valid' in laporan_struk.columns and laporan_struk['tanggal_valid'].dtype == bool else True

    laporan_struk_filtered = laporan_struk[
        (laporan_struk['total_transaksi'] <= 500000) &
        (laporan_struk['total_qty'] <= 50) &
        cond_date &
        (laporan_struk['nama_toko'].str.len() >= 5) &
        (laporan_struk['nama_toko'].str.contains(r'[A-Za-z]{5,}', regex=True, na=False)) &
        (~laporan_struk['nama_toko'].str.contains(
            r'Penjualan|Distribus|\d{8,}|^Shop$|^Tong$|^Trre$|Deeok|Distrab|Duplikat|Ptoomfa|Mamy Poko|Opnstd|Apaaja|Yangaku|Higuna|Wiguna',
            case=False, regex=True, na=False
        ))
    ].copy()

    # Sinkronisasi visualisasi item agar memotong row outliers toko noise
    valid_filenames = laporan_struk_filtered['filename'].unique()
    df_clean_valid = df_clean[df_clean['filename'].isin(valid_filenames)].copy()

    # ==========================================
    # VISUALISASI GRID GRAFIK PERILAKU PASAR
    # ==========================================
    st.subheader("📈 Analisis Kecenderungan Pasar dan Audit Finansial")
    
    col_chartA, col_chartB = st.columns(2)
    
    with col_chartA:
        st.markdown("**A. Distribusi Komponen Data Transaksi**")
        df_harga = df_clean_valid[df_clean_valid['harga_satuan'] > 0] if not df_clean_valid.empty else df_clean[df_clean['harga_satuan'] > 0]
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
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}rb'))
        axes[0].legend()

        # Chart 2: Distribusi jumlah barang (QTY)
        df_qty_source = df_clean_valid if not df_clean_valid.empty else df_clean
        jumlah_counts = df_qty_source['jumlah_barang'].value_counts().sort_index().head(10) if 'jumlah_barang' in df_qty_source.columns else pd.Series([0], index=[1])
        
        bars1 = axes[1].bar(
            jumlah_counts.index.astype(str),
            jumlah_counts.values,
            color='mediumseagreen',
            alpha=0.85,
            edgecolor='white'
        )
        axes[1].bar_label(bars1, padding=3)
        axes[1].set_title('Distribusi Jumlah Barang per Baris Transaksi', fontweight='bold')
        axes[1].set_xlabel('Jumlah Barang')
        axes[1].set_ylabel('Frekuensi')

        plt.suptitle('BQ3 — Distribusi Data Transaksi', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col_chartB:
        st.markdown("**B. Segmentasi Item Berdasarkan Rentang Harga**")
        kat_order = [
            'Sangat Murah (<=5rb)',
            'Murah (5-20rb)',
            'Sedang (20-50rb)',
            'Mahal (50-100rb)',
            'Sangat Mahal (>100rb)'
        ]
        df_kat_source = df_clean_valid if not df_clean_valid.empty else df_clean
        kat_counts = df_kat_source['kategori_harga'].value_counts().reindex(kat_order, fill_value=0)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors_kat = ['steelblue', 'mediumseagreen', 'orange', 'tomato', 'mediumpurple']

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

    # Baris baru melebar ke bawah untuk meninjau audit 10 Struk Teratas
    st.markdown("**C. Pengeluaran Teratas Berdasarkan Berkas Struk Nota Belanja Valid**")
    
    top_struk = (
        laporan_struk_filtered
        .sort_values('total_transaksi', ascending=False)
        .head(10)
        .sort_values('total_transaksi')
    )

    if not top_struk.empty:
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
    else:
        st.warning("Tidak ada data transaksi valid untuk ditampilkan pada grafik pengeluaran teratas.")

    st.divider()

    # ==========================================
    # LAPORAN TABEL TERSTRUKTUR DIBUNGKUS EXPANDER
    # ==========================================
    with st.expander("📂 Lihat Lembar Dokumen Transaksi Terstruktur (Database Hasil Agregasi OCR Final)"):
        st.write("Daftar 10 baris teratas nota belanja hasil pembacaan database terstruktur bersih:")
        
        kolom_tabel = ['filename', 'nama_toko', 'tanggal_clean', 'jumlah_item', 'total_qty', 'total_transaksi', 'estimasi_laba']
        kolom_tabel_ada = [c for c in kolom_tabel if c in laporan_struk_filtered.columns]
        
        st.dataframe(
            laporan_struk_filtered[kolom_tabel_ada].sort_values('total_transaksi', ascending=False).head(10),
            use_container_width=True
        )

    st.divider()

    # ==========================================
    # INSIGHT RESMI BISNIS DARI NOTEBOOK
    # ==========================================
    st.subheader("💡 Insight Analisis Pertanyaan Bisnis 3")
    st.markdown("""
    Data transaksi hasil OCR berhasil diolah menjadi laporan terstruktur setelah melalui proses *cleaning* dan *feature engineering*.

    Sekitar **69% item** berada pada kategori **Murah (5–20rb)** dan **Sedang (20–50rb)** dengan median harga satuan **Rp 15.145**, mencerminkan pola belanja kebutuhan sehari-hari. Hanya sebagian kecil item masuk kategori Mahal dan Sangat Mahal, masing-masing 9 item.

    **Mayoritas transaksi bersifat satuan (1 unit per baris)**, bukan grosir. Dataset mencakup berbagai jenis toko mulai dari minimarket, warung, kafe, hingga apotek.

    Data dapat diagregasi menjadi laporan ringkas per struk yang memuat nama toko, tanggal, total item, total transaksi, dan estimasi laba. Dengan asumsi margin 20%, sistem dapat langsung menghasilkan estimasi laba tanpa input harga beli manual, sehingga praktis untuk pembukuan sederhana pelaku UMKM.

    > **Catatan Teknis Penulisan:** Beberapa nama toko dan nilai total transaksi masih mengandung *noise* OCR residual. Normalisasi nama toko lebih lanjut dapat dilakukan menggunakan teknik *fuzzy matching* pada tahap pengembangan berikutnya.
    """)

# Footer Global
st.divider()
st.caption("Copyright © 2026 | Proyek NOPI AI Analysis Dashboard")

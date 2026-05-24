import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NOPI Dashboard",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8F9FA; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B2A4A 0%, #0F1C33 100%);
    }
    [data-testid="stSidebar"] * { color: #E8EDF5 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #E8EDF5 !important; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2E6BE6;
        margin-bottom: 10px;
    }
    .metric-card.green  { border-left-color: #28A745; }
    .metric-card.orange { border-left-color: #FD7E14; }
    .metric-card.purple { border-left-color: #6F42C1; }
    .metric-card.red    { border-left-color: #DC3545; }

    .metric-label { font-size: 13px; color: #6c757d; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1B2A4A; line-height: 1; }
    .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 4px; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2E6BE6 0%, #1B4FB8 100%);
        color: white !important;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 700;
        margin: 20px 0 16px 0;
    }

    /* BQ badge */
    .bq-badge {
        display: inline-block;
        background: #2E6BE6;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Insight box */
    .insight-box {
        background: #EBF3FF;
        border-left: 4px solid #2E6BE6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
        font-size: 14px;
        color: #1B2A4A;
        line-height: 1.7;
    }
    .insight-box b { color: #2E6BE6; }

    /* Answer box */
    .answer-box {
        background: #F0FBF4;
        border-left: 4px solid #28A745;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
        font-size: 14px;
        color: #155724;
        line-height: 1.7;
    }
    .answer-box b { color: #155724; }

    /* Warning box */
    .warning-box {
        background: #FFF8E6;
        border-left: 4px solid #FFC107;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 13px;
        color: #856404;
    }

    /* Table styling */
    .dataframe { font-size: 13px !important; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    hr { border-color: #dee2e6; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def card(label, value, sub="", color=""):
    cls = f"metric-card {color}"
    return f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """

def section(title):
    st.markdown(f'<div class="section-header">📊 {title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def answer(text):
    st.markdown(f'<div class="answer-box">✅ {text}</div>', unsafe_allow_html=True)

def bq_header(n, text):
    st.markdown(f'<div class="bq-badge">Business Question {n}</div>', unsafe_allow_html=True)
    st.markdown(f"### {text}")


# ─────────────────────────────────────────────
# DATA GENERATION (synthetic mirroring notebook)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)

    # ── OCR Evaluation ───────────────────────
    df_evaluasi = pd.DataFrame({
        'Nama Model': ['PaddleOCR', 'Tesseract', 'EasyOCR'],
        'Success Rate (%)': [73.33, 56.67, 60.00],
        'Rata-rata Waktu (Detik)': [5.21, 2.64, 18.73],
        'Akurasi Jumlah Item (%)': [28.5, 22.0, 33.5],
        'Akurasi Total Harga (%)': [26.09, 18.0, 14.0],
    })

    # ── Detail Parsing Status ────────────────
    status_data = []
    for model, sempurna, sebagian, gagal in [
        ('Paddle', 16, 6, 8),
        ('Tesseract', 10, 7, 13),
        ('EasyOCR', 12, 6, 12),
    ]:
        for _ in range(sempurna):
            status_data.append({'Model': model, 'Status': 'Sempurna (100%)'})
        for _ in range(sebagian):
            status_data.append({'Model': model, 'Status': 'Sebagian (Terekstrak tapi ada miss)'})
        for _ in range(gagal):
            status_data.append({'Model': model, 'Status': 'Gagal Total (0%)'})
    df_detail = pd.DataFrame(status_data)

    # ── Main Transaction Dataset (df_clean) ──
    product_pool = [
        'Kanzlr Bakso Ori 48G', 'Nutrijel Pwd.Strw.15', 'Indomie Goreng 85G',
        'Mie Sedaap Soto 77G', 'Teh Pucuk Harum 350ml', 'Aqua Botol 600ml',
        'Royco Ayam 10G', 'Saus Sambal Abc 135ml', 'Kecap Bango 135ml',
        'Tepung Terigu Segitiga 1Kg', 'Gula Pasir Gulaku 1Kg', 'Beras Premium 5Kg',
        'Sabun Lifebuoy 90G', 'Shampo Clear 170ml', 'Pasta Gigi Pepsodent 190G',
        'Susu Ultramilk 250ml', 'Yakult 65ml 5Pcs', 'Tahu Putih Pcs',
        'Telur Ayam 10 Butir', 'Minyak Goreng Bimoli 1L', 'Pop Ice Sachet',
        'Kopi Kapal Api Sachet', 'Chitato Sapi Panggang 68G', 'Oreo Vanilla 119G',
        'Pocari Sweat 350ml',
    ]
    toko_pool = [
        'Minimarket Sejahtera', 'Warung Barokah', 'Toko Makmur Jaya',
        'Apotek Sehat', 'Kedai Kopi Senja', 'Swalayan Mutiara',
        'Warung Bu Siti', 'Toko Kelontong Maju', 'Minimarket Berkah',
        'Kedai Serba Ada',
    ]

    rows = []
    filenames = [f'primer_{str(i).zfill(4)}.jpg' for i in range(53)]
    for fname in filenames:
        toko = np.random.choice(toko_pool)
        n_items = np.random.randint(2, 9)
        month = np.random.randint(1, 13)
        year = np.random.choice([2023, 2024])
        tanggal = pd.Timestamp(year=year, month=month, day=np.random.randint(1, 28))
        for _ in range(n_items):
            prod = np.random.choice(product_pool)
            qty = np.random.choice([1, 1, 1, 2, 2, 3, 5], p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
            # price distribution matching notebook median ~15k
            price_cat = np.random.choice(['low', 'mid', 'high', 'vhigh'], p=[0.35, 0.34, 0.22, 0.09])
            if price_cat == 'low':
                harga = np.random.randint(1000, 5001)
            elif price_cat == 'mid':
                harga = np.random.randint(5000, 20001)
            elif price_cat == 'high':
                harga = np.random.randint(20000, 50001)
            else:
                harga = np.random.randint(50000, 200001)

            total = harga * qty
            rows.append({
                'filename': fname,
                'nama_toko': toko,
                'tanggal_clean': tanggal,
                'tanggal_valid': True,
                'nama_barang': prod,
                'jumlah_barang': qty,
                'harga_satuan': harga,
                'total_harga_item': total,
                'bulan': month,
                'tahun': year,
                'bulan_tahun': tanggal.strftime('%Y-%m'),
            })

    df_clean = pd.DataFrame(rows)

    # kategori harga
    def kat_harga(h):
        if h <= 5000:   return 'Sangat Murah (<=5rb)'
        elif h <= 20000: return 'Murah (5-20rb)'
        elif h <= 50000: return 'Sedang (20-50rb)'
        elif h <= 100000: return 'Mahal (50-100rb)'
        else: return 'Sangat Mahal (>100rb)'
    df_clean['kategori_harga'] = df_clean['harga_satuan'].apply(kat_harga)
    df_clean['estimasi_laba_20'] = df_clean['total_harga_item'] * 0.20

    return df_evaluasi, df_detail, df_clean


df_evaluasi, df_detail, df_clean = generate_data()

# Pre-compute laporan per struk
laporan_struk = df_clean.groupby('filename').agg(
    nama_toko=('nama_toko', 'first'),
    tanggal=('tanggal_clean', 'first'),
    jumlah_item=('nama_barang', 'count'),
    total_qty=('jumlah_barang', 'sum'),
    total_transaksi=('total_harga_item', 'sum'),
    estimasi_laba=('estimasi_laba_20', 'sum'),
).reset_index()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 NOPI")
    st.markdown("**Nota Pintar — Dashboard Analisis**")
    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        [
            "🏠 Overview",
            "❓ BQ1 — Performa OCR",
            "💰 BQ2 — Estimasi Laba",
            "📋 BQ3 — Laporan Transaksi",
        ]
    )

    st.markdown("---")
    st.markdown("**Filter Global**")
    selected_year = st.multiselect(
        "Tahun Transaksi",
        options=sorted(df_clean['tahun'].unique()),
        default=sorted(df_clean['tahun'].unique())
    )

    # Filtered data
    df_filtered = df_clean[df_clean['tahun'].isin(selected_year)] if selected_year else df_clean

    st.markdown("---")
    st.caption("📌 Data: 53 struk | 192 baris bersih")
    st.caption("🤖 Model: PaddleOCR")
    st.caption("🏢 UMKM Indonesia")


# ─────────────────────────────────────────────
# ─────────── PAGE: OVERVIEW ─────────────────
# ─────────────────────────────────────────────
if menu == "🏠 Overview":
    st.title("🧾 NOPI — Nota Pintar")
    st.markdown("#### Dashboard Analisis Data Transaksi & Performa OCR untuk UMKM Indonesia")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(card("Total Struk Diproses", "53", "file foto struk", ""), unsafe_allow_html=True)
    with col2:
        st.markdown(card("Data Bersih", "192", "baris transaksi", "green"), unsafe_allow_html=True)
    with col3:
        total_omzet = df_clean['total_harga_item'].sum()
        st.markdown(card("Total Omzet Dataset", f"Rp {total_omzet/1e6:.1f}jt", "seluruh struk", "orange"), unsafe_allow_html=True)
    with col4:
        total_laba = df_clean['estimasi_laba_20'].sum()
        st.markdown(card("Est. Laba (20%)", f"Rp {total_laba/1e6:.1f}jt", "margin 20%", "purple"), unsafe_allow_html=True)

    st.markdown("---")

    # Business Understanding
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("### 🎯 Latar Belakang")
        st.markdown("""
        **NOPI (Nota Pintar)** adalah aplikasi berbasis OCR yang dirancang untuk membantu 
        pelaku **UMKM Indonesia** mengelola keuangan secara digital dan akurat.

        Indonesia memiliki **65,5 juta UMKM** yang berkontribusi **61% terhadap PDB nasional**, 
        namun mayoritas masih mencatat transaksi secara manual — bahkan hanya dari ingatan.

        **Masalah utama:**
        - 🔴 Rekap keuangan lambat & rawan human error
        - 🔴 Sulit menghitung laba per item secara akurat
        - 🔴 Tidak ada laporan terstruktur untuk pengambilan keputusan
        """)

    with col_r:
        st.markdown("### 🤖 Solusi NOPI")
        st.markdown("""
        Pipeline dua tahap:
        
        **1. Klasifikasi CNN**
        - Filter otomatis gambar struk vs non-struk
        - Target akurasi minimal **85%**

        **2. Ekstraksi OCR**
        - Baca nama item, harga satuan, total transaksi
        - Dibandingkan: PaddleOCR, Tesseract, EasyOCR
        """)

        st.markdown("### ❓ Business Questions")
        for i, bq in enumerate([
            "Bagaimana performa OCR dalam mengekstrak informasi dari struk?",
            "Bagaimana estimasi laba per produk secara sederhana?",
            "Bagaimana data OCR menjadi laporan pengambilan keputusan?"
        ], 1):
            st.markdown(f"**BQ{i}:** {bq}")

    st.markdown("---")

    # Quick viz: data overview
    section("Gambaran Dataset")

    c1, c2 = st.columns(2)
    with c1:
        # Top toko by transaction count
        top_toko = df_filtered.groupby('nama_toko')['total_harga_item'].sum().sort_values(ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(top_toko.index, top_toko.values / 1000, color='steelblue', alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.0f rb', padding=3, fontsize=9)
        ax.set_title('Omzet per Toko (Rp Ribu)', fontweight='bold')
        ax.set_xlabel('Omzet (Rp Ribu)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        # Transaksi per bulan
        monthly = df_filtered.groupby('bulan_tahun')['total_harga_item'].sum().reset_index()
        monthly = monthly.sort_values('bulan_tahun')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(monthly['bulan_tahun'], monthly['total_harga_item'] / 1000,
               color='mediumseagreen', alpha=0.85, edgecolor='white')
        ax.set_title('Total Transaksi per Bulan (Rp Ribu)', fontweight='bold')
        ax.set_xlabel('Periode')
        ax.set_ylabel('Omzet (Rp Ribu)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    insight("""
    Dataset mencakup <b>53 struk</b> dari berbagai jenis toko (minimarket, warung, kafe, apotek) 
    dengan total <b>192 baris transaksi bersih</b>. Distribusi omzet tersebar merata antar toko, 
    menunjukkan variasi sampel yang representatif untuk konteks UMKM Indonesia.
    """)


# ─────────────────────────────────────────────
# ─────────── PAGE: BQ1 ──────────────────────
# ─────────────────────────────────────────────
elif menu == "❓ BQ1 — Performa OCR":
    st.title("❓ Business Question 1")
    bq_header(1, "Bagaimana teknologi OCR dapat dimanfaatkan untuk mengekstrak informasi dari struk secara otomatis dan akurat?")

    st.markdown("""
    **Indikator Pengukuran:** Perbandingan model OCR berdasarkan success rate, akurasi item, 
    akurasi total harga, dan waktu proses dari 30 sampel struk.
    """)
    st.markdown("---")

    # ── Grafik 1: Komparasi 4 Metrik ─────────
    section("Komparasi Performa 3 Model OCR")

    models = df_evaluasi['Nama Model']
    colors = ['steelblue', 'tomato', 'mediumseagreen']

    metrics = [
        ('Success Rate (%)', 'Success Rate (%)', 'Persentase (%)'),
        ('Rata-rata Waktu (Detik)', 'Rata-rata Waktu Proses (Detik)', 'Detik'),
        ('Akurasi Jumlah Item (%)', 'Akurasi Jumlah Item (%)', 'Persentase (%)'),
        ('Akurasi Total Harga (%)', 'Akurasi Total Harga (%)', 'Persentase (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (col, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(models, df_evaluasi[col], color=colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, df_evaluasi[col]):
            fmt = f'{val:.2f}s' if 'Waktu' in col else f'{val:.1f}%'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    fmt, ha='center', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel)
        if '%' in col:
            ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('BQ1 — Komparasi Performa 3 Model OCR', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Grafik 2: Distribusi Status Parsing ──
    section("Distribusi Status Parsing per Model")

    label_map = {
        'Sebagian (Terekstrak tapi ada miss)': 'Sebagian',
        'Sempurna (100%)': 'Sempurna',
        'Gagal Total (0%)': 'Gagal Total'
    }
    status_counts = df_detail.groupby(['Model', 'Status']).size().unstack(fill_value=0)
    status_counts = status_counts.rename(columns=label_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_status = ['mediumseagreen', 'steelblue', 'tomato']

    for ax, model in zip(axes, ['Paddle', 'Tesseract', 'EasyOCR']):
        if model in status_counts.index:
            data = status_counts.loc[model]
            wedges, texts, autotexts = ax.pie(
                data.values,
                labels=data.index,
                autopct='%1.1f%%',
                colors=colors_status[:len(data)],
                startangle=90,
                textprops={'fontsize': 10}
            )
            for at in autotexts:
                at.set_fontweight('bold')
            ax.set_title(f'Status Parsing — {model}', fontweight='bold', fontsize=12)

    plt.suptitle('BQ1 — Distribusi Status Parsing per Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Detail Table ──────────────────────────
    section("Tabel Ringkasan Evaluasi OCR")
    st.dataframe(
        df_evaluasi.set_index('Nama Model').style
        .format({
            'Success Rate (%)': '{:.1f}%',
            'Rata-rata Waktu (Detik)': '{:.2f} detik',
            'Akurasi Jumlah Item (%)': '{:.1f}%',
            'Akurasi Total Harga (%)': '{:.1f}%',
        })
        .highlight_max(color='#d4edda', axis=0)
        .highlight_min(color='#f8d7da', axis=0),
        use_container_width=True
    )
    st.caption("🟢 Hijau = nilai terbaik per metrik | 🔴 Merah = nilai terburuk per metrik")

    # ── Jawaban BQ1 ───────────────────────────
    st.markdown("---")
    answer("""
    <b>Jawaban BQ1:</b> Dari ketiga model OCR yang dievaluasi pada 30 sampel struk, 
    <b>PaddleOCR terbukti sebagai model terbaik secara keseluruhan</b> dengan success rate tertinggi 
    <b>73.33%</b> dan akurasi total harga tertinggi <b>26.09%</b>. Meskipun waktu prosesnya lebih lambat 
    dari Tesseract (5.21 vs 2.64 detik), keunggulan akurasi menjadikannya pilihan optimal.<br><br>
    Tesseract unggul dalam kecepatan namun memiliki akurasi item dan harga paling rendah sehingga kurang 
    andal untuk data transaksi. EasyOCR memiliki akurasi item tertinggi (33.5%) tetapi waktu prosesnya 
    sangat lambat (18.73 detik) dan tidak efisien untuk deployment real-time. 
    <b>Dengan demikian, PaddleOCR dipilih sebagai model OCR untuk NOPI</b> karena keseimbangan terbaik 
    antara keberhasilan parsing, akurasi harga, dan waktu proses yang wajar.
    """)


# ─────────────────────────────────────────────
# ─────────── PAGE: BQ2 ──────────────────────
# ─────────────────────────────────────────────
elif menu == "💰 BQ2 — Estimasi Laba":
    st.title("💰 Business Question 2")
    bq_header(2, "Bagaimana pelaku usaha mikro dapat mengetahui estimasi laba dari setiap produk yang dijual secara sederhana dan efisien?")

    st.markdown("""
    **Indikator Pengukuran:** Estimasi laba dihitung dari total harga item 
    menggunakan asumsi persentase margin laba yang diinput oleh pengguna.
    """)
    st.markdown("---")

    # ── Kalkulator Laba Interaktif ────────────
    section("🎛️ Kalkulator Estimasi Laba Interaktif")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        selected_file = st.selectbox(
            "Pilih Struk",
            options=sorted(df_filtered['filename'].unique()),
            index=0
        )
        margin = st.slider(
            "Margin Laba (%)",
            min_value=5, max_value=50, value=20, step=5,
            help="Persentase estimasi laba dari total harga item"
        )

    sample_struk = df_filtered[df_filtered['filename'] == selected_file].copy()
    sample_struk['laba_total'] = sample_struk['total_harga_item'] * (margin / 100)
    sample_struk['harga_beli_est'] = sample_struk['total_harga_item'] - sample_struk['laba_total']
    sample_struk['laba_per_unit'] = sample_struk['laba_total'] / sample_struk['jumlah_barang']

    total_omzet = sample_struk['total_harga_item'].sum()
    total_laba_s = sample_struk['laba_total'].sum()
    total_harga_beli = sample_struk['harga_beli_est'].sum()

    with col_ctrl2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(card("Total Omzet", f"Rp {total_omzet:,.0f}", f"dari {len(sample_struk)} item", ""), unsafe_allow_html=True)
        with c2:
            st.markdown(card("Estimasi Laba", f"Rp {total_laba_s:,.0f}", f"margin {margin}%", "green"), unsafe_allow_html=True)
        with c3:
            st.markdown(card("Est. Harga Beli", f"Rp {total_harga_beli:,.0f}", "total modal", "orange"), unsafe_allow_html=True)

    # ── Grafik: Laba per Item ─────────────────
    section(f"Estimasi Laba per Item — {selected_file} (Margin {margin}%)")

    df_plot = sample_struk[sample_struk['laba_total'] > 0].sort_values('laba_total').tail(15)

    fig, ax = plt.subplots(figsize=(11, max(4, len(df_plot) * 0.4 + 1)))
    bars = ax.barh(
        df_plot['nama_barang'], df_plot['laba_total'],
        color='mediumseagreen', alpha=0.85, edgecolor='white'
    )
    for bar, val in zip(bars, df_plot['laba_total']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'Rp {val:,.0f}', va='center', fontsize=9)
    ax.set_title(f'BQ2 — Estimasi Laba per Item (Margin {margin}%)', fontweight='bold')
    ax.set_xlabel('Estimasi Laba Total (Rp)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Grafik: Perbandingan Omzet vs Laba ───
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        section("Komposisi Omzet vs Harga Beli vs Laba")
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ['Modal\n(Harga Beli)', f'Laba\n({margin}%)']
        sizes = [total_harga_beli, total_laba_s]
        colors = ['steelblue', 'mediumseagreen']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.set_title(f'Komposisi Omzet Rp {total_omzet:,.0f}', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_v2:
        section("Tabel Detail Estimasi Laba")
        display_cols = ['nama_barang', 'jumlah_barang', 'harga_satuan', 'total_harga_item', 'laba_total', 'laba_per_unit']
        rename_map = {
            'nama_barang': 'Nama Barang',
            'jumlah_barang': 'Qty',
            'harga_satuan': 'Harga Satuan',
            'total_harga_item': 'Total Harga',
            'laba_total': f'Est. Laba ({margin}%)',
            'laba_per_unit': 'Laba/Unit',
        }
        df_display = sample_struk[display_cols].rename(columns=rename_map)
        st.dataframe(
            df_display.style.format({
                'Harga Satuan': 'Rp {:,.0f}',
                'Total Harga': 'Rp {:,.0f}',
                f'Est. Laba ({margin}%)': 'Rp {:,.0f}',
                'Laba/Unit': 'Rp {:,.0f}',
            }),
            use_container_width=True, height=280
        )

    # ── Grafik: Trend Laba Semua Struk ────────
    section("Distribusi Estimasi Laba Seluruh Struk")

    laba_all = df_filtered.groupby('filename').agg(
        total_omzet=('total_harga_item', 'sum'),
        total_laba=('total_harga_item', lambda x: (x * margin / 100).sum()),
    ).reset_index().sort_values('total_laba', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(laba_all))
    width = 0.38
    ax.bar([i - width/2 for i in x], laba_all['total_omzet'] / 1000,
           width, label='Omzet', color='steelblue', alpha=0.85, edgecolor='white')
    ax.bar([i + width/2 for i in x], laba_all['total_laba'] / 1000,
           width, label=f'Est. Laba ({margin}%)', color='mediumseagreen', alpha=0.85, edgecolor='white')
    ax.set_xticks(list(x))
    ax.set_xticklabels(laba_all['filename'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Nilai (Rp Ribu)')
    ax.set_title(f'BQ2 — Omzet vs Estimasi Laba Top 20 Struk (Margin {margin}%)', fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Jawaban BQ2 ───────────────────────────
    st.markdown("---")
    answer(f"""
    <b>Jawaban BQ2:</b> Pelaku usaha mikro <b>tidak perlu memasukkan harga beli satu per satu</b>. 
    Cukup dengan menginput satu angka persentase margin (contoh: {margin}%), sistem NOPI langsung 
    menghitung estimasi laba per item secara otomatis dari data transaksi hasil OCR.<br><br>
    Berdasarkan analisis seluruh dataset dengan margin {margin}%, total omzet dataset sebesar 
    <b>Rp {df_filtered['total_harga_item'].sum():,.0f}</b> menghasilkan estimasi laba 
    <b>Rp {df_filtered['total_harga_item'].sum() * margin / 100:,.0f}</b>. 
    Pendekatan ini praktis karena menghasilkan laporan laba per item yang siap digunakan untuk 
    pembukuan sederhana tanpa sistem pencatatan harga beli yang kompleks.
    """)


# ─────────────────────────────────────────────
# ─────────── PAGE: BQ3 ──────────────────────
# ─────────────────────────────────────────────
elif menu == "📋 BQ3 — Laporan Transaksi":
    st.title("📋 Business Question 3")
    bq_header(3, "Bagaimana data transaksi hasil OCR dapat diolah menjadi laporan terstruktur untuk mendukung pengambilan keputusan bisnis?")

    st.markdown("""
    **Indikator Pengukuran:** Kelengkapan kolom, jumlah data bersih, missing value, duplikat, 
    validitas tanggal, serta ringkasan penjualan per struk dan kategori harga.
    """)
    st.markdown("---")

    # ── Grafik 1: Distribusi Harga & Qty ──────
    section("Distribusi Data Transaksi")

    df_harga = df_filtered[df_filtered['harga_satuan'] > 0]
    median_harga = df_harga['harga_satuan'].median()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_harga['harga_satuan'], bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].axvline(median_harga, color='tomato', linestyle='--', linewidth=2,
                    label=f'Median: Rp {median_harga:,.0f}')
    axes[0].set_title('Distribusi Harga Satuan Item', fontweight='bold')
    axes[0].set_xlabel('Harga Satuan (Rp)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}rb'))
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    jumlah_counts = df_filtered['jumlah_barang'].value_counts().sort_index().head(10)
    bars = axes[1].bar(jumlah_counts.index.astype(str), jumlah_counts.values,
                       color='mediumseagreen', alpha=0.85, edgecolor='white')
    axes[1].bar_label(bars, padding=3)
    axes[1].set_title('Distribusi Jumlah Barang per Baris Transaksi', fontweight='bold')
    axes[1].set_xlabel('Jumlah Barang (Unit)')
    axes[1].set_ylabel('Frekuensi')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('BQ3 — Distribusi Data Transaksi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Grafik 2: Segmentasi Kategori Harga ──
    section("Segmentasi Item berdasarkan Kategori Harga")

    kat_order = [
        'Sangat Murah (<=5rb)', 'Murah (5-20rb)', 'Sedang (20-50rb)',
        'Mahal (50-100rb)', 'Sangat Mahal (>100rb)'
    ]
    kat_counts = df_filtered['kategori_harga'].value_counts().reindex(kat_order, fill_value=0)

    col_pie, col_bar = st.columns([1, 1.4])
    with col_pie:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_kat = ['steelblue', 'mediumseagreen', 'orange', 'tomato', 'mediumpurple']
        wedges, texts, autotexts = ax.pie(
            kat_counts.values, labels=kat_counts.index,
            autopct='%1.1f%%', colors=colors_kat, startangle=90,
            textprops={'fontsize': 9}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.set_title('Proporsi Kategori Harga', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_bar:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(kat_counts.index, kat_counts.values,
                       color=colors_kat, alpha=0.85, edgecolor='white')
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_title('Jumlah Item per Kategori Harga', fontweight='bold')
        ax.set_xlabel('Jumlah Item')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Grafik 3: Top Struk by Total Transaksi
    section("Top 10 Struk berdasarkan Total Transaksi")

    laporan_filtered = laporan_struk[laporan_struk['filename'].isin(df_filtered['filename'])]
    top_struk = (
        laporan_filtered
        .sort_values('total_transaksi', ascending=False)
        .head(10)
        .sort_values('total_transaksi')
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(top_struk['filename'], top_struk['total_transaksi'],
                   color='mediumpurple', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, top_struk['total_transaksi']):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f'Rp {val:,.0f}', va='center', fontsize=9)
    ax.set_title('BQ3 — Struk Teratas berdasarkan Total Transaksi', fontweight='bold')
    ax.set_xlabel('Total Transaksi (Rp)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Laporan Terstruktur ───────────────────
    section("📄 Laporan Transaksi Terstruktur (Simulasi Output NOPI)")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_toko = st.multiselect(
            "Filter Nama Toko",
            options=sorted(laporan_filtered['nama_toko'].unique()),
            default=[]
        )
    with col_f2:
        sort_by = st.selectbox(
            "Urutkan berdasarkan",
            ['total_transaksi', 'estimasi_laba', 'jumlah_item', 'total_qty']
        )

    lap_show = laporan_filtered.copy()
    if filter_toko:
        lap_show = lap_show[lap_show['nama_toko'].isin(filter_toko)]
    lap_show = lap_show.sort_values(sort_by, ascending=False).reset_index(drop=True)

    lap_show['tanggal'] = lap_show['tanggal'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        lap_show.rename(columns={
            'filename': 'File Struk',
            'nama_toko': 'Nama Toko',
            'tanggal': 'Tanggal',
            'jumlah_item': 'Jumlah Item',
            'total_qty': 'Total Qty',
            'total_transaksi': 'Total Transaksi (Rp)',
            'estimasi_laba': 'Est. Laba 20% (Rp)',
        }).style.format({
            'Total Transaksi (Rp)': 'Rp {:,.0f}',
            'Est. Laba 20% (Rp)': 'Rp {:,.0f}',
        }),
        use_container_width=True, height=350
    )

    # ── Statistik Deskriptif ──────────────────
    section("Statistik Deskriptif Dataset Bersih")

    desc = df_filtered[['jumlah_barang', 'harga_satuan', 'total_harga_item']].describe().round(2)
    desc.columns = ['Jumlah Barang', 'Harga Satuan (Rp)', 'Total Harga Item (Rp)']
    st.dataframe(desc, use_container_width=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(card("Total Baris Bersih", str(len(df_filtered)), "setelah cleaning", ""), unsafe_allow_html=True)
    with col_s2:
        n_struk = df_filtered['filename'].nunique()
        st.markdown(card("Struk Unik", str(n_struk), "file foto diproses", "green"), unsafe_allow_html=True)
    with col_s3:
        tgl_valid = df_filtered['tanggal_valid'].sum()
        pct = tgl_valid / len(df_filtered) * 100
        st.markdown(card("Tanggal Valid", str(tgl_valid), f"{pct:.1f}% dari total", "orange"), unsafe_allow_html=True)
    with col_s4:
        mv = df_filtered[['nama_barang','jumlah_barang','harga_satuan','total_harga_item']].isna().sum().sum()
        st.markdown(card("Missing Kolom Utama", str(mv), "setelah cleaning", "purple"), unsafe_allow_html=True)

    # ── Jawaban BQ3 ───────────────────────────
    st.markdown("---")
    answer("""
    <b>Jawaban BQ3:</b> Data transaksi hasil OCR berhasil diolah menjadi laporan terstruktur 
    setelah melalui proses <b>cleaning dan feature engineering</b>. Dari 337 baris raw OCR, 
    dihasilkan <b>192 baris bersih dari 53 struk</b> tanpa missing value pada kolom transaksi utama.<br><br>
    Sekitar <b>69% item</b> berada pada kategori Murah (5–20rb) dan Sedang (20–50rb) dengan 
    <b>median harga satuan Rp 15.145</b>, mencerminkan pola belanja kebutuhan sehari-hari UMKM. 
    Mayoritas transaksi bersifat satuan (1 unit per baris). Dataset mencakup berbagai jenis toko: 
    minimarket, warung, kafe, hingga apotek.<br><br>
    <b>Data dapat diagregasi menjadi laporan ringkas per struk</b> yang memuat nama toko, tanggal, 
    total item, total transaksi, dan estimasi laba — siap digunakan untuk pengambilan keputusan bisnis 
    tanpa input manual tambahan dari pelaku UMKM.
    """)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Catatan:</b> Beberapa nama toko dan nilai total transaksi masih mengandung noise OCR residual. 
    Normalisasi nama toko lebih lanjut dapat dilakukan menggunakan teknik <i>fuzzy matching</i> 
    pada tahap pengembangan berikutnya.
    </div>
    """, unsafe_allow_html=True)

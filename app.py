import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Nota Pintar (NOPI)",
    page_icon="📄",
    layout="wide"
)

# --- FUNGSI LOAD DATA (Caching agar cepat) ---
@st.cache_data
def load_data():
    # Pastikan file ini ada di folder 'data/' di GitHub kamu
    try:
        df = pd.read_csv('data/df_all.csv')
        return df
    except:
        # Data cadangan jika file tidak ditemukan saat pertama kali setup
        return pd.DataFrame({
            'label': ['struk']*1014 + ['non_struk']*1014,
            'width': np.random.normal(689, 754, 2028).clip(180, 6720),
            'aspect_ratio': np.random.uniform(0.2, 1.5, 2028)
        })

df_all = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigasi Dashboard")
st.sidebar.info("Proyek: Nota Pintar (NOPI)\nLayer: DS-2 (Analysis & Insight)")
menu = st.sidebar.radio("Pilih Analisis:", ["Ringkasan Data", "Analisis Geometris", "Performa Model AI"])

# --- MENU 1: RINGKASAN DATA ---
if menu == "Ringkasan Data":
    st.header("📊 Ringkasan Distribusi Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Jumlah Data per Kelas")
        fig, ax = plt.subplots()
        sns.countplot(data=df_all, x='label', palette='viridis', ax=ax)
        st.pyplot(fig)
        st.write("**Insight:** Dataset seimbang antara kelas Struk dan Non-Struk, mencegah model menjadi bias.")

    with col2:
        st.subheader("Persentase Komposisi")
        fig, ax = plt.subplots()
        counts = df_all['label'].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FF5722'])
        st.pyplot(fig)
        st.write("**Insight:** Distribusi 50:50 ideal untuk pelatihan model klasifikasi biner.")

# --- MENU 2: ANALISIS GEOMETRIS ---
elif menu == "Analisis Geometris":
    st.header("📈 Analisis Resolusi & Bentuk Gambar")
    
    tab1, tab2 = st.tabs(["Tren Lebar (Width)", "Aspect Ratio (Pola Bentuk)"])
    
    with tab1:
        st.subheader("Tren Distribusi Lebar Gambar")
        sorted_width = np.sort(df_all['width'].values)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sorted_width, color='orange', linewidth=2)
        ax.set_ylabel("Width (Pixel)")
        ax.set_xlabel("Urutan Data")
        st.pyplot(fig)
        st.warning("**Insight Outlier:** Lonjakan tajam di ujung grafik menunjukkan adanya gambar dengan resolusi ekstrem (>2000px) yang perlu di-resize.")

    with tab2:
        st.subheader("Perbandingan Bentuk (Aspect Ratio)")
        fig, ax = plt.subplots()
        sns.boxplot(x='label', y='aspect_ratio', data=df_all, palette='Pastel1', ax=ax)
        st.pyplot(fig)
        st.write("**Insight Geometris:** Label 'Struk' secara konsisten memiliki aspect ratio rendah (Portrait), menjadikannya ciri khas utama pembeda.")

# --- MENU 3: PERFORMA MODEL AI ---
elif menu == "Performa Model AI":
    st.header("🎯 Evaluasi Performa Model CNN")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Akurasi", "95%", "Target: 90%")
    col_m2.metric("Precision", "100%", "Sangat Stabil")
    col_m3.metric("Recall", "90%", "Perlu Perbaikan")

    st.divider()
    
    col_cm, col_txt = st.columns([2, 1])
    
    with col_cm:
        # Simulasi Confusion Matrix sesuai data sebelumnya
        cm = np.array([[48, 0], [5, 47]])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Struk', 'Struk'])
        fig, ax = plt.subplots()
        disp.plot(cmap='Blues', ax=ax)
        st.pyplot(fig)
        
    with col_txt:
        st.subheader("Kesimpulan Performa")
        st.write("""
        - **False Positive (0):** Model tidak pernah salah mengira gambar biasa sebagai struk.
        - **False Negative (5):** Ada 5 struk yang gagal dikenali, kemungkinan karena resolusi terlalu rendah.
        - **Rekomendasi:** Tambahkan data struk yang variatif untuk meningkatkan Recall.
        """)
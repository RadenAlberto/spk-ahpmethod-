import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- CONFIG & TEMA ---
st.set_page_config(page_title="AHP Pertagas Niaga", layout="wide")

# Styling dikit biar gak kaku
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-card { background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- LOGIKA MATEMATIKA AHP ---
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}

def get_weights(matrix):
    col_sum = matrix.sum(axis=0)
    norm_matrix = matrix / col_sum
    weights = norm_matrix.mean(axis=1)
    
    # Hitung Konsistensi
    n = len(matrix)
    if n <= 2:
        return weights, 0.0
    
    lampda_max = np.mean((matrix @ weights) / weights)
    ci = (lampda_max - n) / (n - 1)
    cr = ci / RI[n]
    return weights, cr

# --- SIDEBAR (Input Data) ---
with st.sidebar:
    st.header("📋 Input Data")
    input_crit = st.text_area("Kriteria (Pisahkan dengan koma):", 
                             "Harga, Kualitas, Layanan, Reputasi")
    input_vend = st.text_area("Vendor (Pisahkan dengan koma):", 
                             "PT PMS, Vendor B, Vendor C")
    
    criteria = [x.strip() for x in input_crit.split(",") if x.strip()]
    vendors = [x.strip() for x in input_vend.split(",") if x.strip()]
    
    st.divider()
    st.info("Cara Pakai: Isi kriteria/vendor di atas, lalu atur perbandingan di tab yang muncul.")

# --- MAIN APP ---
st.title("⛽ SPK Pemilihan Vendor")
st.caption("Aplikasi perhitungan AHP untuk PT Pertamina Pertagas Niaga")

if len(criteria) < 2 or len(vendors) < 2:
    st.warning("Tambahkan minimal 2 kriteria dan 2 vendor di sidebar.")
    st.stop()

tabs = st.tabs(["1. Bobot Kriteria", "2. Perbandingan Vendor", "3. Hasil Akhir"])

# --- TAB 1: KRITERIA ---
with tabs[0]:
    st.subheader("Seberapa penting kriteria ini dibanding yang lain?")
    n = len(criteria)
    c_matrix = np.ones((n, n))
    
    # Form input yang lebih manusiawi (pake kolom)
    for i in range(n):
        for j in range(i + 1, n):
            col1, col2 = st.columns([2, 3])
            with col1:
                st.write(f"**{criteria[i]}** vs **{criteria[j]}**")
            with col2:
                val = st.select_slider(
                    "Skala Kepentingan",
                    options=[9, 7, 5, 3, 1, 1/3, 1/5, 1/7, 1/9],
                    value=1,
                    format_func=lambda x: f"Penting {int(x)}" if x >= 1 else f"Kebalikan (1/{int(1/x)})",
                    key=f"c_{i}_{j}"
                )
                c_matrix[i, j] = val
                c_matrix[j, i] = 1 / val
    
    c_weights, c_cr = get_weights(c_matrix)
    
    # Tampilan hasil ringkas
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Hasil Bobot Kriteria")
        df_crit = pd.DataFrame({"Kriteria": criteria, "Bobot": c_weights})
        st.dataframe(df_crit.sort_values("Bobot", ascending=False).style.format({"Bobot": "{:.2%}"}))
    with col_b:
        st.write("#### Cek Konsistensi")
        if c_cr < 0.1:
            st.success(f"Konsisten (CR = {c_cr:.2f})")
        else:
            st.error(f"Gak Konsisten (CR = {c_cr:.2f}). Coba atur ulang nilainya.")

# --- TAB 2: VENDOR ---
v_weights_matrix = [] # Simpan bobot tiap vendor per kriteria

with tabs[1]:
    st.subheader("Bandingkan Vendor pada masing-masing Kriteria")
    
    for k_idx, k_name in enumerate(criteria):
        with st.expander(f"Berdasarkan Kriteria: {k_name}", expanded=(k_idx == 0)):
            m_v = np.ones((len(vendors), len(vendors)))
            for i in range(len(vendors)):
                for j in range(i + 1, len(vendors)):
                    val_v = st.select_slider(
                        f"{vendors[i]} vs {vendors[j]}",
                        options=[9, 7, 5, 3, 1, 1/3, 1/5, 1/7, 1/9],
                        value=1,
                        key=f"v_{k_idx}_{i}_{j}"
                    )
                    m_v[i, j] = val_v
                    m_v[j, i] = 1 / val_v
            
            v_w, v_cr = get_weights(m_v)
            v_weights_matrix.append(v_w)
            st.caption(f"Konsistensi {k_name}: {v_cr:.2f}")

# --- TAB 3: HASIL ---
with tabs[2]:
    st.header("🏁 Rekomendasi Vendor")
    
    if len(v_weights_matrix) == len(criteria):
        # Hitung skor akhir (Matrix Multiplication)
        final_scores = np.array(v_weights_matrix).T @ c_weights
        
        results = pd.DataFrame({
            "Vendor": vendors,
            "Skor Akhir": final_scores
        }).sort_values("Skor Akhir", ascending=False)
        
        # Visualisasi
        fig = px.bar(results, x="Skor Akhir", y="Vendor", orientation='h', 
                     title="Ranking Vendor", color="Skor Akhir", 
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Detail Skor")
        st.table(results.style.format({"Skor Akhir": "{:.4f}"}))
        
        st.success(f"Kesimpulan: **{results.iloc[0]['Vendor']}** adalah pilihan terbaik.")
    else:
        st.info("Selesaikan perbandingan di Tab 1 dan 2 dulu ya.")
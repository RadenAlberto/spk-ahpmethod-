import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# -------------------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------------------
st.set_page_config(
    page_title="SPK AHP - PT Pertamina Pertagas Niaga",
    page_icon="⛽",
    layout="wide"
)

# Tabel Random Index (RI) Standar Saaty
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# -------------------------------------------------------------
# FUNGSI PERHITUNGAN AHP
# -------------------------------------------------------------
def hitung_ahp(matrix):
    n = matrix.shape[0]
    col_sum = matrix.sum(axis=0)
    col_sum_safe = np.where(col_sum == 0, 1e-9, col_sum)
    
    # Normalisasi & Bobot (Eigenvector)
    norm_matrix = matrix / col_sum_safe
    weights = norm_matrix.mean(axis=1)
    
    # Uji Konsistensi
    if n <= 2:
        return {
            "matrix": matrix,
            "norm_matrix": norm_matrix,
            "weights": weights,
            "lambda_max": float(n),
            "ci": 0.0,
            "cr": 0.0,
            "konsisten": True
        }
    
    wsv = matrix @ weights
    cv = wsv / np.where(weights == 0, 1e-9, weights)
    lambda_max = float(np.mean(cv))
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    
    return {
        "matrix": matrix,
        "norm_matrix": norm_matrix,
        "weights": weights,
        "lambda_max": lambda_max,
        "ci": ci,
        "cr": cr,
        "konsisten": cr <= 0.10
    }

def render_input_berpasangan(items, prefix_key):
    """Komponen form perbandingan berpasangan yang aman & intuitif"""
    n = len(items)
    matrix = np.ones((n, n), dtype=float)
    
    skala_saaty = {
        1: "1 - Sama penting",
        2: "2 - Nilai antara (sedikit mendekati)",
        3: "3 - Sedikit lebih penting",
        4: "4 - Nilai antara",
        5: "5 - Jelas lebih penting",
        6: "6 - Nilai antara",
        7: "7 - Sangat jelas lebih penting",
        8: "8 - Nilai antara",
        9: "9 - Mutlak lebih penting"
    }

    st.markdown("Pilih entitas yang **lebih dominan/penting** dan tentukan bobot nilainya:")
    
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = st.columns([1, 2])
            with c1:
                pilihan = st.radio(
                    f"Dominansi #{i+1}-{j+1}:",
                    options=[items[i], items[j]],
                    horizontal=True,
                    key=f"rad_{prefix_key}_{i}_{j}"
                )
            with c2:
                nilai = st.select_slider(
                    "Tingkat Kepentingan:",
                    options=list(skala_saaty.keys()),
                    format_func=lambda x: skala_saaty[x],
                    value=1,
                    key=f"sld_{prefix_key}_{i}_{j}"
                )
            
            if pilihan == items[i]:
                matrix[i, j] = float(nilai)
                matrix[j, i] = 1.0 / float(nilai)
            else:
                matrix[i, j] = 1.0 / float(nilai)
                matrix[j, i] = float(nilai)
            st.divider()
            
    return matrix

# -------------------------------------------------------------
# SIDEBAR / NAVIGASI
# -------------------------------------------------------------
with st.sidebar:
    st.title("⛽ Pertagas Niaga")
    st.subheader("Konfigurasi Kasus")
    
    kriteria_default = "Harga Sewa, Kondisi Armada, Kecepatan Layanan, Track Record"
    input_kriteria = st.text_area("Daftar Kriteria (Pisahkan koma):", kriteria_default, height=90)
    kriteria = [k.strip() for k in input_kriteria.split(",") if k.strip()]
    
    vendor_default = "PT PMS, Vendor B, Vendor C, Vendor D"
    input_vendor = st.text_area("Daftar Vendor/Alternatif (Pisahkan koma):", vendor_default, height=90)
    vendor = [v.strip() for v in input_vendor.split(",") if v.strip()]
    
    st.caption("Pastikan minimal 2 kriteria dan 2 vendor terisi.")

# -------------------------------------------------------------
# HEADER APLIKASI
# -------------------------------------------------------------
st.title("Sistem Pendukung Keputusan — Metode AHP")
st.markdown("**Studi Kasus:** Pemilihan Vendor Kendaraan Operasional PT Pertamina Pertagas Niaga")
st.write("---")

# Validasi Jumlah Data
if len(kriteria) < 2 or len(vendor) < 2:
    st.error("⚠️ Masukkan minimal 2 kriteria dan 2 vendor di panel sebelah kiri untuk memulai.")
    st.stop()

# -------------------------------------------------------------
# WIDGET TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Bobot Kriteria", 
    "2. Penilaian Alternatif", 
    "3. Hasil Akhir & Ranking", 
    "4. Uji Sensitivitas"
])

# -------------------------------------------------------------
# TAB 1: BOBOT KRITERIA
# -------------------------------------------------------------
with tab1:
    st.subheader("⚖️ Perbandingan Berpasangan Antar-Kriteria")
    matriks_kriteria = render_input_berpasangan(kriteria, "kriteria")
    res_kriteria = hitung_ahp(matriks_kriteria)
    
    # Ringkasan Metrik
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("λ Max", f"{res_kriteria['lambda_max']:.4f}")
    m2.metric("CI (Indeks Konsistensi)", f"{res_kriteria['ci']:.4f}")
    m3.metric("CR (Rasio Konsistensi)", f"{(res_kriteria['cr'] * 100):.2f}%")
    with m4:
        st.markdown("**Status Uji:**")
        if res_kriteria["konsisten"]:
            st.success("✅ KONSISTEN (CR ≤ 10%)")
        else:
            st.error("❌ TIDAK KONSISTEN (CR > 10%)")
            st.caption("Disarankan mengevaluasi kembali nilai perbandingan kriteria di atas.")

    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        st.markdown("##### Matriks Perbandingan Kriteria")
        df_mat = pd.DataFrame(matriks_kriteria, index=kriteria, columns=kriteria)
        st.dataframe(df_mat.style.format("{:.3f}"), use_container_width=True)
        
    with col_k2:
        st.markdown("##### Bobot Prioritas Kriteria ($W_C$)")
        df_bobot_kriteria = pd.DataFrame({
            "Kriteria": kriteria,
            "Bobot": res_kriteria["weights"],
            "Persentase": res_kriteria["weights"] * 100
        }).sort_values("Bobot", ascending=False)
        
        fig_crit = px.pie(
            df_bobot_kriteria, 
            names="Kriteria", 
            values="Bobot", 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_crit.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_crit, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: PENILAIAN ALTERNATIF
# -------------------------------------------------------------
bobot_vendor_per_kriteria = {}

with tab2:
    st.subheader("🏢 Perbandingan Alternatif Vendor untuk Tiap Kriteria")
    subtabs = st.tabs([f"Kriteria: {k}" for k in kriteria])
    
    for idx, k_name in enumerate(kriteria):
        with subtabs[idx]:
            st.markdown(f"##### Evaluasi Vendor berdasarkan: **{k_name}**")
            m_v = render_input_berpasangan(vendor, f"v_{idx}")
            res_v = hitung_ahp(m_v)
            bobot_vendor_per_kriteria[k_name] = res_v["weights"]
            
            c_info1, c_info2 = st.columns([1, 2])
            with c_info1:
                st.metric(f"CR ({k_name})", f"{(res_v['cr'] * 100):.2f}%")
                if res_v["konsisten"]:
                    st.success("Konsisten")
                else:
                    st.warning("Perlu evaluasi (CR > 10%)")
            with c_info2:
                df_res_v = pd.DataFrame({
                    "Vendor": vendor,
                    "Bobot Lokal": res_v["weights"],
                    "Kontribusi (%)": res_v["weights"] * 100
                }).sort_values("Bobot Lokal", ascending=False)
                st.dataframe(df_res_v.style.format({"Bobot Lokal": "{:.4f}", "Kontribusi (%)": "{:.2f}%"}), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: HASIL AKHIR & REKOMENDASI
# -------------------------------------------------------------
with tab3:
    st.subheader("🏆 Sintesis & Peringkat Akhir")
    
    # Matriks Alternatif (m x n)
    W_A = np.column_stack([bobot_vendor_per_kriteria[k] for k in kriteria])
    W_C = res_kriteria["weights"]
    
    # Skor Global = W_A * W_C
    skor_akhir = W_A @ W_C
    
    df_hasil = pd.DataFrame({
        "Vendor": vendor,
        "Skor Akhir": skor_akhir,
        "Persentase": skor_akhir * 100
    }).sort_values("Skor Akhir", ascending=False).reset_index(drop=True)
    df_hasil["Peringkat"] = [f"Rank {i+1}" for i in range(len(df_hasil))]
    
    pemenang = df_hasil.iloc[0]["Vendor"]
    skor_pemenang = df_hasil.iloc[0]["Skor Akhir"]
    
    st.success(f"🎯 **Rekomendasi Utama:** Vendor terbaik adalah **{pemenang}** dengan skor akhir **{skor_pemenang:.4f}** ({(skor_pemenang*100):.2f}%).")
    
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown("##### Tabel Ranking Rekomendasi")
        st.dataframe(
            df_hasil[["Peringkat", "Vendor", "Skor Akhir", "Persentase"]].style
            .format({"Skor Akhir": "{:.4f}", "Persentase": "{:.2f}%"}),
            use_container_width=True
        )
        
        # Tombol Download Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_hasil.to_excel(writer, index=False, sheet_name='Hasil_AHP')
        
        st.download_button(
            label="📥 Unduh Hasil Ranking (.xlsx)",
            data=buf.getvalue(),
            file_name="Hasil_AHP_Pertagas_Niaga.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    with col_h2:
        st.markdown("##### Visualisasi Komparasi Nilai Akhir")
        fig_bar = px.bar(
            df_hasil.sort_values("Skor Akhir", ascending=True),
            x="Skor Akhir",
            y="Vendor",
            orientation='h',
            text_auto='.4f',
            color="Skor Akhir",
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar Chart Profil Vendor
    st.write("---")
    st.markdown("##### 📊 Profil Kekuatan Tiap Vendor (Radar Chart)")
    fig_radar = go.Figure()
    for idx_v, nama_v in enumerate(vendor):
        nilai_radar = W_A[idx_v, :].tolist()
        nilai_radar += [nilai_radar[0]]  # Menutup loop radar
        kriteria_radar = kriteria + [kriteria[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=nilai_radar,
            theta=kriteria_radar,
            fill='toself',
            name=nama_v
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, np.max(W_A) * 1.15])),
        showlegend=True,
        height=400,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: UJI SENSITIVITAS
# -------------------------------------------------------------
with tab4:
    st.subheader("🔬 Analisis Sensitivitas (What-If Analysis)")
    st.write("Uji ketahanan keputusan dengan mengubah bobot kriteria secara fleksibel:")
    
    sim_weights = []
    cols = st.columns(len(kriteria))
    for idx, k_name in enumerate(kriteria):
        with cols[idx]:
            val_sim = st.slider(
                f"{k_name}",
                min_value=0.0,
                max_value=1.0,
                value=float(res_kriteria["weights"][idx]),
                step=0.05,
                key=f"sim_slider_{idx}"
            )
            sim_weights.append(val_sim)
            
    total_sim = sum(sim_weights)
    norm_sim_weights = np.array(sim_weights) / (total_sim if total_sim > 0 else 1e-9)
    
    # Hitung skor baru
    skor_simulasi = W_A @ norm_sim_weights
    
    df_sim = pd.DataFrame({
        "Vendor": vendor,
        "Skor Awal AHP": skor_akhir,
        "Skor Pasca Simulasi": skor_simulasi
    }).sort_values("Skor Pasca Simulasi", ascending=False)
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(name='Skor Awal', x=df_sim['Vendor'], y=df_sim['Skor Awal AHP'], marker_color='#94A3B8'))
    fig_sim.add_trace(go.Bar(name='Skor Simulasi', x=df_sim['Vendor'], y=df_sim['Skor Pasca Simulasi'], marker_color='#0284C7'))
    fig_sim.update_layout(barmode='group', height=350, margin=dict(t=20, b=20, l=20, r=20))
    
    st.plotly_chart(fig_sim, use_container_width=True)
    st.dataframe(df_sim.style.format({"Skor Awal AHP": "{:.4f}", "Skor Pasca Simulasi": "{:.4f}"}), use_container_width=True)
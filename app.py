import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# ==========================================
# PAGE CONFIGURATION & ENTERPRISE STYLING
# ==========================================
st.set_page_config(
    page_title="SPK AHP — PT Pertamina Pertagas Niaga",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS (Pertamina Theme: Blue, Red, Clean Grey)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 50%, #005691 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(10, 37, 64, 0.2);
    }
    
    .metric-card {
        background: #FFFFFF;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .badge-consistent {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #84E1BC;
    }
    
    .badge-inconsistent {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #F8B4B4;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SAATY CONSTANTS & MATH FUNCTIONS
# ==========================================
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SAATY_SCALE = {
    1: "1 - Sama Penting (Equal)",
    2: "2 - Sedikit Mendekati Lebih Penting",
    3: "3 - Sedikit Lebih Penting (Moderate)",
    4: "4 - Mendekati Jelas Lebih Penting",
    5: "5 - Jelas Lebih Penting (Strong)",
    6: "6 - Mendekati Sangat Jelas Lebih Penting",
    7: "7 - Sangat Jelas Lebih Penting (Very Strong)",
    8: "8 - Mendekati Mutlak Lebih Penting",
    9: "9 - Mutlak Lebih Penting (Extreme)"
}

def calculate_ahp(matrix):
    """
    Hitung Bobot Prioritas, Lambda Max, CI, dan CR
    """
    n = matrix.shape[0]
    col_sum = matrix.sum(axis=0)
    col_sum_safe = np.where(col_sum == 0, 1e-10, col_sum)
    norm_matrix = matrix / col_sum_safe
    weights = norm_matrix.mean(axis=1)
    
    # Weighted Sum Vector (WSV)
    wsv = matrix @ weights
    # Consistency Vector (CV)
    cv = wsv / np.where(weights == 0, 1e-10, weights)
    lambda_max = float(np.mean(cv))
    
    ci = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = (ci / ri) if (ri > 0 and n > 2) else 0.0
    
    return {
        "matrix": matrix,
        "norm_matrix": norm_matrix,
        "weights": weights,
        "lambda_max": lambda_max,
        "ci": ci,
        "cr": cr,
        "is_consistent": cr <= 0.10
    }

def render_pairwise_form(items, group_key, default_matrix=None):
    """
    UI Form Interaktif Skala Saaty (User Friendly tanpa input desimal 0.111)
    """
    n = len(items)
    matrix = np.ones((n, n), dtype=float)
    
    if default_matrix is not None and default_matrix.shape == (n, n):
        matrix = default_matrix.copy()

    with st.expander(f"📝 Formulir Perbandingan Berpasangan ({len(items)} Entitas)", expanded=True):
        st.caption("Pilih item mana yang lebih dominan/penting beserta tingkat kepentingannya berdasarkan Skala Saaty (1-9):")
        
        for i in range(n):
            for j in range(i + 1, n):
                item_a = items[i]
                item_b = items[j]
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    pref = st.radio(
                        f"Dominansi #{i+1}-{j+1}:",
                        options=[item_a, item_b],
                        horizontal=True,
                        key=f"pref_{group_key}_{i}_{j}"
                    )
                with c2:
                    val = st.select_slider(
                        f"Tingkat Kepentingan:",
                        options=list(SAATY_SCALE.keys()),
                        format_func=lambda x: SAATY_SCALE[x],
                        value=1,
                        key=f"scale_{group_key}_{i}_{j}"
                    )
                
                if pref == item_a:
                    matrix[i, j] = float(val)
                    matrix[j, i] = 1.0 / float(val)
                else:
                    matrix[i, j] = 1.0 / float(val)
                    matrix[j, i] = float(val)
                st.divider()
                
    return matrix

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Pertamina_Logo.svg/2560px-Pertamina_Logo.svg.png", width=180)
    st.markdown("### ⚙️ Pengaturan Evaluasi")
    
    # Input Kriteria
    default_criteria = "Cost (Harga Sewa), Quality (Kondisi Armada), Service (Kecepatan Layanan), Reputation (Track Record)"
    raw_criteria = st.text_area("Daftar Kriteria (Pisahkan dengan koma):", default_criteria, height=90)
    criteria_list = [c.strip() for c in raw_criteria.split(",") if c.strip()]
    
    # Input Alternatif
    default_vendors = "PT PMS, Vendor B, Vendor C, Vendor D"
    raw_vendors = st.text_area("Daftar Vendor/Alternatif (Pisahkan dengan koma):", default_vendors, height=90)
    vendor_list = [v.strip() for v in raw_vendors.split(",") if v.strip()]

    st.markdown("---")
    st.info("💡 **AHP Rule:** Pastikan rasio konsistensi $CR \\le 10\\%$ (0.10) agar keputusan valid.")

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 1.5px; opacity: 0.8; text-transform: uppercase;">
        Sistem Pendukung Keputusan (SPK)
    </div>
    <div style="font-size: 1.9rem; font-weight: 800; margin: 4px 0;">
        Analytical Hierarchy Process (AHP)
    </div>
    <div style="font-size: 1rem; opacity: 0.9;">
        Studi Kasus: Pemilihan Vendor Kendaraan Operasional — PT Pertamina Pertagas Niaga
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs Workflow
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Bobot Kriteria", 
    "2️⃣ Evaluasi Vendor", 
    "🏆 Hasil Akhir & Rekomendasi", 
    "🔬 Analisis Sensitivitas"
])

# ==========================================
# TAB 1: BOBOT KRITERIA
# ==========================================
with tab1:
    st.markdown("### 🎯 Langkah 1: Perbandingan Berpasangan Antar-Kriteria")
    
    if len(criteria_list) < 2:
        st.error("Minimal harus ada 2 kriteria untuk perbandingan AHP.")
    else:
        crit_matrix = render_pairwise_form(criteria_list, "criteria")
        crit_res = calculate_ahp(crit_matrix)
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("λ Max (Eigenvalue)", f"{crit_res['lambda_max']:.4f}")
        with col2:
            st.metric("CI (Consistency Index)", f"{crit_res['ci']:.4f}")
        with col3:
            st.metric("CR (Consistency Ratio)", f"{(crit_res['cr']*100):.2f}%")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if crit_res['is_consistent']:
                st.markdown('<span class="badge-consistent">✅ KONSISTEN (CR ≤ 10%)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-inconsistent">⚠️ TIDAK KONSISTEN (CR > 10%)</span>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Matriks Perbandingan Kriteria")
            st.dataframe(pd.DataFrame(crit_matrix, index=criteria_list, columns=criteria_list).style.format("{:.3f}"), use_container_width=True)
            
            with st.expander("Lihat Matriks Normalisasi"):
                st.dataframe(pd.DataFrame(crit_res["norm_matrix"], index=criteria_list, columns=criteria_list).style.format("{:.4f}"), use_container_width=True)

        with col_right:
            st.markdown("#### Distribusi Bobot Prioritas Kriteria")
            df_crit_weights = pd.DataFrame({
                "Kriteria": criteria_list,
                "Bobot": crit_res["weights"],
                "Persentase": crit_res["weights"] * 100
            }).sort_values(by="Bobot", ascending=True)
            
            fig_pie = px.pie(
                df_crit_weights, 
                names="Kriteria", 
                values="Bobot", 
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(criteria_list))
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 2: EVALUASI ALTERNATIF / VENDOR
# ==========================================
vendor_weights_per_criteria = {}
all_vendors_consistent = True

with tab2:
    st.markdown("### 🏢 Langkah 2: Perbandingan Alternatif Vendor per Kriteria")
    st.caption("Nilai keunggulan masing-masing vendor ditinjau dari tiap kriteria yang ada.")
    
    if len(vendor_list) < 2:
        st.error("Minimal harus ada 2 vendor/alternatif untuk perbandingan.")
    else:
        v_tabs = st.tabs([f"📌 {crit}" for crit in criteria_list])
        
        for idx, crit in enumerate(criteria_list):
            with v_tabs[idx]:
                st.markdown(f"#### Perbandingan Vendor Berdasarkan: **{crit}**")
                v_matrix = render_pairwise_form(vendor_list, f"vendor_{idx}")
                v_res = calculate_ahp(v_matrix)
                vendor_weights_per_criteria[crit] = v_res["weights"]
                
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.metric("CR Kriteria Ini", f"{(v_res['cr']*100):.2f}%")
                with c2:
                    if v_res['is_consistent']:
                        st.markdown('<br><span class="badge-consistent">✅ Konsisten</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<br><span class="badge-inconsistent">⚠️ Tidak Konsisten</span>', unsafe_allow_html=True)
                        all_vendors_consistent = False
                with c3:
                    df_v = pd.DataFrame({
                        "Vendor": vendor_list,
                        "Skor Lokal": v_res["weights"],
                        "Persentase": v_res["weights"] * 100
                    }).sort_values(by="Skor Lokal", ascending=False)
                    st.dataframe(df_v.style.format({"Skor Lokal": "{:.4f}", "Persentase": "{:.2f}%"}), use_container_width=True)

# ==========================================
# TAB 3: HASIL AKHIR & REKOMENDASI
# ==========================================
with tab3:
    st.markdown("### 🏆 Hasil Akhir Sintesis & Rekomendasi Keputusan")
    
    if len(vendor_weights_per_criteria) == len(criteria_list):
        # Matriks Alternatif (m x n)
        W_A = np.column_stack([vendor_weights_per_criteria[c] for c in criteria_list])
        W_C = crit_res["weights"]
        
        # Skor Akhir Global: W_A * W_C
        final_scores = W_A @ W_C
        
        df_final = pd.DataFrame({
            "Vendor": vendor_list,
            "Nilai Preferensi Global": final_scores,
            "Persentase": final_scores * 100
        }).sort_values(by="Nilai Preferensi Global", ascending=False).reset_index(drop=True)
        df_final["Peringkat"] = [f"Peringkat {i+1}" for i in range(len(df_final))]
        
        best_vendor = df_final.iloc[0]["Vendor"]
        best_score = df_final.iloc[0]["Nilai Preferensi Global"]

        # Alert Box Juara
        st.success(f"""
        ### 🌟 Rekomendasi Utama: **{best_vendor}**
        Berdasarkan perhitungan multi-kriteria AHP, **{best_vendor}** menempati prioritas tertinggi dengan skor preferensi akhir **{best_score:.4f}** ({best_score*100:.2f}%).
        """)
        
        col_rank1, col_rank2 = st.columns([1, 1])
        
        with col_rank1:
            st.markdown("#### Tabel Peringkat Akhir")
            st.dataframe(
                df_final[["Peringkat", "Vendor", "Nilai Preferensi Global", "Persentase"]].style
                .format({"Nilai Preferensi Global": "{:.4f}", "Persentase": "{:.2f}%"})
                .background_gradient(subset=["Nilai Preferensi Global"], cmap="Blues"),
                use_container_width=True
            )
            
            # Export to Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_final.to_excel(writer, index=False, sheet_name='Hasil_AHP')
            
            st.download_button(
                label="📥 Unduh Laporan Rekapitulasi (Excel)",
                data=buffer.getvalue(),
                file_name="Laporan_AHP_Pertamina_Pertagas_Niaga.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col_rank2:
            st.markdown("#### Visualisasi Komparasi Nilai Akhir")
            fig_bar = px.bar(
                df_final.sort_values(by="Nilai Preferensi Global", ascending=True),
                x="Nilai Preferensi Global",
                y="Vendor",
                orientation='h',
                text_auto='.4f',
                color="Nilai Preferensi Global",
                color_continuous_scale="Teal"
            )
            fig_bar.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Matriks Kontribusi Tiap Kriteria ke Tiap Vendor
        st.markdown("---")
        st.markdown("#### 📊 Profil Radar Kontribusi Vendor per Kriteria")
        
        fig_radar = go.Figure()
        for v_idx, vendor in enumerate(vendor_list):
            scores_radar = W_A[v_idx, :].tolist()
            scores_radar += [scores_radar[0]] # Close loop
            crit_radar = criteria_list + [criteria_list[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=scores_radar,
                theta=crit_radar,
                fill='toself',
                name=vendor
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, np.max(W_A)*1.1])),
            showlegend=True,
            height=420,
            margin=dict(t=30, b=30, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# TAB 4: ANALISIS SENSITIVITAS
# ==========================================
with tab4:
    st.markdown("### 🔬 Analisis Sensitivitas (What-If Analysis)")
    st.caption("Uji ketahanan keputusan dengan mengubah bobot kriteria secara dinamis secara real-time.")
    
    if len(vendor_weights_per_criteria) == len(criteria_list):
        st.markdown("#### Simulasikan Perubahan Bobot Kriteria:")
        
        simulated_weights = []
        cols = st.columns(len(criteria_list))
        
        for idx, crit in enumerate(criteria_list):
            with cols[idx]:
                w_val = st.slider(
                    f"{crit}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=float(crit_res["weights"][idx]), 
                    step=0.05,
                    key=f"sim_{idx}"
                )
                simulated_weights.append(w_val)
        
        total_w = sum(simulated_weights)
        if total_w == 0:
            total_w = 1e-10
        norm_sim_weights = np.array(simulated_weights) / total_w
        
        # Hitung skor simulasi
        sim_final_scores = W_A @ norm_sim_weights
        
        df_sim = pd.DataFrame({
            "Vendor": vendor_list,
            "Skor Awal": final_scores,
            "Skor Simulasi": sim_final_scores
        }).sort_values(by="Skor Simulasi", ascending=False)
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(name='Skor Asli AHP', x=df_sim['Vendor'], y=df_sim['Skor Awal'], marker_color='#94A3B8'))
        fig_sim.add_trace(go.Bar(name='Skor Pasca Simulasi', x=df_sim['Vendor'], y=df_sim['Skor Simulasi'], marker_color='#0284C7'))
        fig_sim.update_layout(barmode='group', height=360, margin=dict(t=20, b=20, l=20, r=20))
        
        st.plotly_chart(fig_sim, use_container_width=True)
        st.dataframe(df_sim.style.format({"Skor Awal": "{:.4f}", "Skor Simulasi": "{:.4f}"}), use_container_width=True)
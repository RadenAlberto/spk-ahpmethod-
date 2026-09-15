import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# ==========================================
# PAGE CONFIG & CUSTOM CSS (MIRIP SCREENSHOT)
# ==========================================
st.set_page_config(
    page_title="SPK AHP — PT Pertamina Pertagas Niaga",
    page_icon="⛽",
    layout="wide"
)

# Custom Styling untuk Action Bar & Tabs
st.markdown("""
<style>
    /* Styling Action Bar Button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #EF4444;
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #DC2626;
        color: white;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Styling Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #2D3748;
        padding-bottom: 4px;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #EF4444 !important;
        border-bottom-color: #EF4444 !important;
    }
    
    /* Card Container */
    .card-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Tabel Random Index (RI) Saaty
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# ==========================================
# SKENARIO & INITIAL DATA
# ==========================================
DEFAULT_CRITERIA = ["Cost (Harga)", "Quality (Armada)", "Service (Layanan)", "Reputation (Track Record)"]
DEFAULT_VENDORS = ["PT PMS", "Vendor B", "Vendor C", "Vendor D"]

SCENARIOS = {
    "Skenario Standar (Kuesioner Pertagas Niaga)": [
        [1.00, 3.00, 2.00, 4.00],
        [0.33, 1.00, 0.50, 2.00],
        [0.50, 2.00, 1.00, 3.00],
        [0.25, 0.50, 0.33, 1.00]
    ],
    "Skenario Fokus Efisiensi Biaya (Cost Dominant)": [
        [1.00, 5.00, 4.00, 6.00],
        [0.20, 1.00, 0.50, 2.00],
        [0.25, 2.00, 1.00, 2.00],
        [0.17, 0.50, 0.50, 1.00]
    ],
    "Skenario Fokus Kualitas & Safety (Quality First)": [
        [1.00, 0.33, 0.50, 2.00],
        [3.00, 1.00, 2.00, 4.00],
        [2.00, 0.50, 1.00, 3.00],
        [0.50, 0.25, 0.33, 1.00]
    ],
    "Skenario Equal (Semua Kriteria Sama Penting)": [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]
}

DEFAULT_VEND_MATRICES = {
    "Cost (Harga)": [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.0, 2.0, 3.0],
        [0.33, 0.5, 1.0, 2.0],
        [0.25, 0.33, 0.5, 1.0]
    ],
    "Quality (Armada)": [
        [1.0, 3.0, 2.0, 4.0],
        [0.33, 1.0, 0.5, 2.0],
        [0.5, 2.0, 1.0, 3.0],
        [0.25, 0.5, 0.33, 1.0]
    ],
    "Service (Layanan)": [
        [1.0, 2.0, 4.0, 3.0],
        [0.5, 1.0, 3.0, 2.0],
        [0.25, 0.33, 1.0, 0.5],
        [0.33, 0.5, 2.0, 1.0]
    ],
    "Reputation (Track Record)": [
        [1.0, 4.0, 3.0, 5.0],
        [0.25, 1.0, 0.5, 2.0],
        [0.33, 2.0, 1.0, 3.0],
        [0.2, 0.5, 0.33, 1.0]
    ]
}

# Inisialisasi Session State
if "criteria" not in st.session_state:
    st.session_state.criteria = DEFAULT_CRITERIA.copy()
if "vendors" not in st.session_state:
    st.session_state.vendors = DEFAULT_VENDORS.copy()
if "crit_matrix" not in st.session_state:
    st.session_state.crit_matrix = pd.DataFrame(
        SCENARIOS["Skenario Standar (Kuesioner Pertagas Niaga)"],
        index=DEFAULT_CRITERIA,
        columns=DEFAULT_CRITERIA
    )
if "vend_matrices" not in st.session_state:
    st.session_state.vend_matrices = {
        c: pd.DataFrame(DEFAULT_VEND_MATRICES[c], index=DEFAULT_VENDORS, columns=DEFAULT_VENDORS)
        for c in DEFAULT_CRITERIA
    }

def hitung_ahp(df_matrix):
    mat = df_matrix.to_numpy(dtype=float)
    n = len(mat)
    col_sum = mat.sum(axis=0)
    col_sum_safe = np.where(col_sum == 0, 1e-9, col_sum)
    norm_mat = mat / col_sum_safe
    weights = norm_mat.mean(axis=1)
    
    if n <= 2:
        return weights, norm_mat, float(n), 0.0, 0.0, True
    
    wsv = mat @ weights
    cv = wsv / np.where(weights == 0, 1e-9, weights)
    lam = float(np.mean(cv))
    ci = (lam - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weights, norm_mat, lam, ci, cr, (cr <= 0.10)

# ==========================================
# HEADER UTAMA
# ==========================================
st.markdown("### ⛽ Implementasi AHP dalam Pemilihan Vendor Kendaraan Operasional")
st.caption("Studi Kasus: PT Pertamina Pertagas Niaga — Multi-Criteria Decision Making (AHP Method)")

# ==========================================
# TOP ACTION TOOLBAR (PERSIS SEPERTI DI GAMBAR)
# ==========================================
t_col1, t_col2, t_col3, t_col4 = st.columns([1.5, 1.5, 1.3, 3])

with t_col1:
    if st.button("📊 Hitung Ulang", type="primary", use_container_width=True):
        st.rerun()

with t_col2:
    with st.popover("➕ Tambah Vendor", use_container_width=True):
        new_v_name = st.text_input("Nama Vendor Baru:")
        if st.button("Simpan Vendor", use_container_width=True):
            if new_v_name and new_v_name not in st.session_state.vendors:
                st.session_state.vendors.append(new_v_name)
                # Update matriks vendor
                for c in st.session_state.criteria:
                    old_df = st.session_state.vend_matrices[c]
                    new_df = pd.DataFrame(1.0, index=st.session_state.vendors, columns=st.session_state.vendors)
                    for r in old_df.index:
                        for col_name in old_df.columns:
                            new_df.loc[r, col_name] = old_df.loc[r, col_name]
                    st.session_state.vend_matrices[c] = new_df
                st.success(f"Vendor '{new_v_name}' berhasil ditambahkan!")
                st.rerun()

with t_col3:
    if st.button("🔄 Reset Data", use_container_width=True):
        st.session_state.criteria = DEFAULT_CRITERIA.copy()
        st.session_state.vendors = DEFAULT_VENDORS.copy()
        st.session_state.crit_matrix = pd.DataFrame(
            SCENARIOS["Skenario Standar (Kuesioner Pertagas Niaga)"],
            index=DEFAULT_CRITERIA,
            columns=DEFAULT_CRITERIA
        )
        st.session_state.vend_matrices = {
            c: pd.DataFrame(DEFAULT_VEND_MATRICES[c], index=DEFAULT_VENDORS, columns=DEFAULT_VENDORS)
            for c in DEFAULT_CRITERIA
        }
        st.rerun()

with t_col4:
    selected_scenario = st.selectbox(
        "Pilih Skenario Bobot...",
        options=list(SCENARIOS.keys()),
        label_visibility="collapsed"
    )
    # Jika skenario berubah, update matriks kriteria
    if st.session_state.get("last_scenario") != selected_scenario:
        st.session_state.last_scenario = selected_scenario
        st.session_state.crit_matrix = pd.DataFrame(
            SCENARIOS[selected_scenario],
            index=st.session_state.criteria,
            columns=st.session_state.criteria
        )
        st.rerun()

# ==========================================
# 4 TABS NAVIGASI (SESUAI GAMBAR)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 1. Data & Penilaian Vendor",
    "🏆 2. Hasil & Rekomendasi",
    "📊 3. Visualisasi & Radar Chart",
    "📐 4. Detail Perhitungan AHP"
])

# HITUNG SEMUA BOBOT TERLEBIH DAHULU (Anti Error State)
crit_weights, crit_norm, c_lam, c_ci, c_cr, c_cons = hitung_ahp(st.session_state.crit_matrix)

all_v_weights = []
all_v_cr = {}
all_v_norm = {}

for c_name in st.session_state.criteria:
    vw, vnorm, vlam, vci, vcr, vcons = hitung_ahp(st.session_state.vend_matrices[c_name])
    all_v_weights.append(vw)
    all_v_cr[c_name] = (vcr, vcons)
    all_v_norm[c_name] = vnorm

W_A = np.column_stack(all_v_weights)
final_scores = W_A @ crit_weights

df_ranking = pd.DataFrame({
    "Vendor": st.session_state.vendors,
    "Nilai Akhir": final_scores,
    "Persentase": final_scores * 100
}).sort_values("Nilai Akhir", ascending=False).reset_index(drop=True)
df_ranking["Peringkat"] = [f"Peringkat {i+1}" for i in range(len(df_ranking))]

# -------------------------------------------------------------
# TAB 1: DATA & PENILAIAN VENDOR
# -------------------------------------------------------------
with tab1:
    st.markdown("#### Matriks Perbandingan Berpasangan Kriteria")
    st.caption("Ubah angka matriks secara langsung (Skala 1-9 Saaty):")
    
    col_k1, col_k2 = st.columns([3, 2])
    with col_k1:
        edited_crit = st.data_editor(
            st.session_state.crit_matrix,
            key="edit_crit_tab1",
            use_container_width=True
        )
        st.session_state.crit_matrix = edited_crit
    with col_k2:
        st.markdown("**Status Konsistensi Kriteria:**")
        if c_cons:
            st.success(f"✅ Konsisten (CR = {c_cr:.4f} ≤ 0.10)")
        else:
            st.error(f"❌ Tidak Konsisten (CR = {c_cr:.4f} > 0.10)")
        
        df_cw_show = pd.DataFrame({
            "Kriteria": st.session_state.criteria,
            "Bobot Prioritas": crit_weights,
            "Persentase": crit_weights * 100
        }).sort_values("Bobot Prioritas", ascending=False)
        st.dataframe(df_cw_show.style.format({"Bobot Prioritas": "{:.4f}", "Persentase": "{:.2f}%"}), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Matriks Penilaian Vendor per Kriteria")
    
    v_subtabs = st.tabs([f"📌 {c}" for c in st.session_state.criteria])
    for idx, c_name in enumerate(st.session_state.criteria):
        with v_subtabs[idx]:
            cv1, cv2 = st.columns([3, 2])
            with cv1:
                edited_vm = st.data_editor(
                    st.session_state.vend_matrices[c_name],
                    key=f"edit_vm_{idx}",
                    use_container_width=True
                )
                st.session_state.vend_matrices[c_name] = edited_vm
            with cv2:
                vcr_val, vcr_stat = all_v_cr[c_name]
                if vcr_stat:
                    st.success(f"✅ Konsisten (CR = {vcr_val:.4f})")
                else:
                    st.warning(f"⚠️ Perlu Evaluasi (CR = {vcr_val:.4f})")
                
                df_vw_show = pd.DataFrame({
                    "Vendor": st.session_state.vendors,
                    "Skor Lokal": all_v_weights[idx],
                    "Kontribusi (%)": all_v_weights[idx] * 100
                }).sort_values("Skor Lokal", ascending=False)
                st.dataframe(df_vw_show.style.format({"Skor Lokal": "{:.4f}", "Kontribusi (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: HASIL & REKOMENDASI
# -------------------------------------------------------------
with tab2:
    pemenang = df_ranking.iloc[0]["Vendor"]
    skor_pemenang = df_ranking.iloc[0]["Nilai Akhir"]
    
    st.success(f"""
    ### 🏆 Rekomendasi Utama: **{pemenang}**
    Berdasarkan perhitungan AHP, **{pemenang}** memperoleh skor tertinggi sebesar **{skor_pemenang:.4f}** ({skor_pemenang*100:.2f}%) dan direkomendasikan sebagai vendor kendaraan operasional terbaik untuk PT Pertamina Pertagas Niaga.
    """)
    
    col_r1, col_r2 = st.columns([3, 2])
    with col_r1:
        st.markdown("#### Tabel Peringkat Rekomendasi")
        st.dataframe(
            df_ranking[["Peringkat", "Vendor", "Nilai Akhir", "Persentase"]].style
            .format({"Nilai Akhir": "{:.4f}", "Persentase": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Download Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_ranking.to_excel(writer, index=False, sheet_name="Hasil_AHP")
        
        st.download_button(
            label="📥 Unduh Laporan Rekomendasi (Excel)",
            data=buffer.getvalue(),
            file_name="Hasil_AHP_Pertagas_Niaga.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    with col_r2:
        st.markdown("#### Kontribusi Skor Vendor")
        fig_bar = px.bar(
            df_ranking.sort_values("Nilai Akhir", ascending=True),
            x="Nilai Akhir",
            y="Vendor",
            orientation="h",
            text_auto=".4f",
            color="Nilai Akhir",
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: VISUALISASI & RADAR CHART
# -------------------------------------------------------------
with tab3:
    st.markdown("#### 📊 Profil Kekuatan Vendor (Radar Chart Multi-Kriteria)")
    
    c_rad1, c_rad2 = st.columns([3, 2])
    with c_rad1:
        fig_radar = go.Figure()
        for idx_v, v_name in enumerate(st.session_state.vendors):
            v_scores = W_A[idx_v, :].tolist()
            v_scores += [v_scores[0]] # Close loop
            crit_loop = st.session_state.criteria + [st.session_state.criteria[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=v_scores,
                theta=crit_loop,
                fill='toself',
                name=v_name
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, np.max(W_A) * 1.15])),
            showlegend=True,
            height=380,
            margin=dict(t=20, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c_rad2:
        st.markdown("#### Distribusi Bobot Kriteria")
        df_cp = pd.DataFrame({"Kriteria": st.session_state.criteria, "Bobot": crit_weights})
        fig_pie = px.pie(df_cp, names="Kriteria", values="Bobot", hole=0.45, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: DETAIL PERHITUNGAN AHP
# -------------------------------------------------------------
with tab4:
    st.markdown("#### 📐 Rincian Matematis & Transparansi Perhitungan AHP")
    
    with st.expander("Langkah 1: Normalisasi Matriks Kriteria & Bobot", expanded=True):
        st.markdown("Matriks Normalisasi ($R_{ij} = \\frac{a_{ij}}{\\sum a_{ij}}$):")
        df_norm_c = pd.DataFrame(crit_norm, index=st.session_state.criteria, columns=st.session_state.criteria)
        st.dataframe(df_norm_c.style.format("{:.4f}"), use_container_width=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("λ Max (Eigenvalue Terbesar)", f"{c_lam:.4f}")
        col_m2.metric("CI (Consistency Index)", f"{c_ci:.4f}")
        col_m3.metric("CR (Consistency Ratio)", f"{c_cr:.4f}")
        st.caption(f"Rumus: $CI = \\frac{{\\lambda_{{max}} - n}}{{n - 1}} = \\frac{{{c_lam:.4f} - {len(st.session_state.criteria)}}}{{{len(st.session_state.criteria) - 1}}} = {c_ci:.4f}$ | $CR = \\frac{{CI}}{{RI}} = \\frac{{{c_ci:.4f}}}{{{RI_TABLE.get(len(st.session_state.criteria), 0.90)}}} = {c_cr:.4f}$")

    with st.expander("Langkah 2: Matriks Sintesis Akhir ($W_A \\times W_C$)", expanded=True):
        st.markdown("Tabel Matriks Bobot Alternatif terhadap Kriteria ($W_A$):")
        df_wa = pd.DataFrame(W_A, index=st.session_state.vendors, columns=st.session_state.criteria)
        st.dataframe(df_wa.style.format("{:.4f}"), use_container_width=True)
        
        st.markdown("Dikalikan dengan Vektor Bobot Kriteria ($W_C$):")
        for i, v_name in enumerate(st.session_state.vendors):
            perkalian_str = " + ".join([f"({W_A[i, j]:.4f} × {crit_weights[j]:.4f})" for j in range(len(st.session_state.criteria))])
            st.latex(f"\\text{{Skor }}_{{{v_name}}} = {perkalian_str} = \\mathbf{{{final_scores[i]:.4f}}}")
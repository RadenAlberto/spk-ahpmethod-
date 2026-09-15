import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AHP VendorSelect — PT Pertamina Pertagas Niaga",
    page_icon="⛽",
    layout="wide"
)

# Tabel Random Index (RI) Saaty
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# ==========================================
# DEFAULT DATA (STUDI KASUS PERTAGAS NIAGA)
# ==========================================
DEFAULT_CRITERIA = ["Cost (Harga)", "Quality (Armada)", "Service (Layanan)", "Reputation (Track Record)"]
DEFAULT_VENDORS = ["PT PMS", "Vendor B", "Vendor C", "Vendor D"]

# Matriks default kriteria
DEFAULT_CRIT_MATRIX = [
    [1.00, 3.00, 2.00, 4.00],
    [0.33, 1.00, 0.50, 2.00],
    [0.50, 2.00, 1.00, 3.00],
    [0.25, 0.50, 0.33, 1.00]
]

# Matriks default vendor per kriteria
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
        DEFAULT_CRIT_MATRIX, 
        index=DEFAULT_CRITERIA, 
        columns=DEFAULT_CRITERIA
    )
if "vend_matrices" not in st.session_state:
    st.session_state.vend_matrices = {
        c: pd.DataFrame(DEFAULT_VEND_MATRICES[c], index=DEFAULT_VENDORS, columns=DEFAULT_VENDORS)
        for c in DEFAULT_CRITERIA
    }

def reset_data():
    st.session_state.criteria = DEFAULT_CRITERIA.copy()
    st.session_state.vendors = DEFAULT_VENDORS.copy()
    st.session_state.crit_matrix = pd.DataFrame(DEFAULT_CRIT_MATRIX, index=DEFAULT_CRITERIA, columns=DEFAULT_CRITERIA)
    st.session_state.vend_matrices = {
        c: pd.DataFrame(DEFAULT_VEND_MATRICES[c], index=DEFAULT_VENDORS, columns=DEFAULT_VENDORS)
        for c in DEFAULT_CRITERIA
    }

# ==========================================
# FUNGSI PERHITUNGAN AHP
# ==========================================
def hitung_ahp(df_matrix):
    mat = df_matrix.to_numpy(dtype=float)
    n = len(mat)
    
    # 1. Normalisasi Matriks & Bobot
    col_sum = mat.sum(axis=0)
    col_sum_safe = np.where(col_sum == 0, 1e-9, col_sum)
    norm_mat = mat / col_sum_safe
    weights = norm_mat.mean(axis=1)
    
    # 2. Konsistensi
    if n <= 2:
        return weights, norm_mat, 0.0, 0.0, True
    
    wsv = mat @ weights
    cv = wsv / np.where(weights == 0, 1e-9, weights)
    lam = float(np.mean(cv))
    ci = (lam - n) / (n - 1)
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    
    return weights, norm_mat, lam, cr, (cr <= 0.10)

# ==========================================
# HEADER UTAMA
# ==========================================
st.title("📊 VendorSelect — Dashboard SPK AHP")
st.caption("Sistem Pendukung Keputusan Pemilihan Vendor Kendaraan Operasional PT Pertamina Pertagas Niaga menggunakan metode **Analytical Hierarchy Process (AHP)**.")

# Baris Tombol Aksi
col_top1, col_top2 = st.columns([6, 1])
with col_top2:
    if st.button("🔄 Reset Data", use_container_width=True):
        reset_data()
        st.rerun()

st.divider()

# ==========================================
# 1. BOBOT & MATRIKS KRITERIA
# ==========================================
st.subheader("1. Matriks Perbandingan Kriteria")
st.write("Edit nilai perbandingan skala Saaty (1-9) langsung pada tabel di bawah:")

c1, c2 = st.columns([3, 2])

with c1:
    edited_crit_df = st.data_editor(
        st.session_state.crit_matrix,
        key="editor_criteria",
        use_container_width=True
    )
    # Sinkronisasi reciprocal otomatis bila diagonal/atas diedit
    st.session_state.crit_matrix = edited_crit_df

with c2:
    c_weights, c_norm, c_lam, c_cr, c_konsisten = hitung_ahp(st.session_state.crit_matrix)
    
    st.markdown("**Hasil Bobot Prioritas Kriteria:**")
    df_cw = pd.DataFrame({
        "Kriteria": st.session_state.criteria,
        "Bobot": c_weights,
        "Persentase": c_weights * 100
    }).sort_values("Bobot", ascending=False)
    
    st.dataframe(
        df_cw.style.format({"Bobot": "{:.4f}", "Persentase": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )
    
    # Status Konsistensi
    if c_konsisten:
        st.success(f"✅ Matriks Konsisten (CR = {c_cr:.4f} ≤ 0.10)")
    else:
        st.error(f"⚠️ Matriks Tidak Konsisten (CR = {c_cr:.4f} > 0.10)")

st.divider()

# ==========================================
# 2. MATRIKS PERBANDINGAN VENDOR PER KRITERIA
# ==========================================
st.subheader("2. Matriks Perbandingan Alternatif Vendor")
st.write("Sesuaikan perbandingan antar-vendor untuk setiap kriteria evaluasi:")

all_v_weights = []

for kriteria_name in st.session_state.criteria:
    with st.expander(f"📌 Perbandingan Vendor Berdasarkan: {kriteria_name}", expanded=True):
        col_v1, col_v2 = st.columns([3, 2])
        
        with col_v1:
            # Ambil data matriks vendor
            current_v_df = st.session_state.vend_matrices.get(
                kriteria_name,
                pd.DataFrame(np.ones((len(st.session_state.vendors), len(st.session_state.vendors))),
                             index=st.session_state.vendors,
                             columns=st.session_state.vendors)
            )
            
            edited_v_df = st.data_editor(
                current_v_df,
                key=f"editor_v_{kriteria_name}",
                use_container_width=True
            )
            st.session_state.vend_matrices[kriteria_name] = edited_v_df
            
        with col_v2:
            v_w, _, _, v_cr, v_konsisten = hitung_ahp(edited_v_df)
            all_v_weights.append(v_w)
            
            df_vw = pd.DataFrame({
                "Vendor": st.session_state.vendors,
                "Skor Lokal": v_w,
                "Kontribusi": v_w * 100
            }).sort_values("Skor Lokal", ascending=False)
            
            st.dataframe(
                df_vw.style.format({"Skor Lokal": "{:.4f}", "Kontribusi": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
            
            if v_konsisten:
                st.caption(f"Status: Konsisten (CR = {v_cr:.4f})")
            else:
                st.caption(f"Status: ⚠️ Tidak Konsisten (CR = {v_cr:.4f})")

st.divider()

# ==========================================
# 3. HASIL AKHIR & REKOMENDASI VENDOR
# ==========================================
st.subheader("3. Hasil Akhir & Rekomendasi Peringkat")

# Matriks Alternatif (m x n) @ Bobot Kriteria (n x 1)
matrix_W_A = np.column_stack(all_v_weights)
final_scores = matrix_W_A @ c_weights

df_hasil = pd.DataFrame({
    "Vendor": st.session_state.vendors,
    "Nilai Preferensi": final_scores,
    "Persentase": final_scores * 100
}).sort_values("Nilai Preferensi", ascending=False).reset_index(drop=True)

df_hasil["Peringkat"] = [f"Rank {i+1}" for i in range(len(df_hasil))]

pemenang = df_hasil.iloc[0]["Vendor"]
skor_pemenang = df_hasil.iloc[0]["Nilai Preferensi"]

# Rekomendasi Box
st.success(f"🏆 **Rekomendasi Vendor Terbaik:** **{pemenang}** terpilih sebagai prioritas nomor 1 dengan skor akhir **{skor_pemenang:.4f}** ({skor_pemenang*100:.2f}%).")

col_res1, col_res2 = st.columns([1, 1])

with col_res1:
    st.markdown("##### Tabel Ranking Vendor")
    st.dataframe(
        df_hasil[["Peringkat", "Vendor", "Nilai Preferensi", "Persentase"]].style
        .format({"Nilai Preferensi": "{:.4f}", "Persentase": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )
    
    # Export to Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_hasil.to_excel(writer, index=False, sheet_name="Hasil_AHP")
    
    st.download_button(
        label="📥 Download Hasil ke Excel",
        data=buffer.getvalue(),
        file_name="Hasil_SPK_AHP_Pertagas_Niaga.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col_res2:
    st.markdown("##### Grafik Komparasi Nilai Akhir")
    fig = px.bar(
        df_hasil.sort_values("Nilai Preferensi", ascending=True),
        x="Nilai Preferensi",
        y="Vendor",
        orientation="h",
        text_auto=".4f",
        color="Nilai Preferensi",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
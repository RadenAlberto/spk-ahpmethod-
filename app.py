import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="AHP Pertamina Pertagas Niaga", page_icon="📊", layout="wide")

RI = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}

def priority(matrix):
    col_sum = matrix.sum(axis=0)
    normalized = matrix / col_sum
    weights = normalized.mean(axis=1)
    return weights, normalized

def consistency(matrix, weights):
    n = len(weights)
    lam = np.mean((matrix @ weights) / weights)
    ci = (lam-n)/(n-1) if n > 2 else 0
    cr = ci/RI[n] if RI[n] else 0
    return lam, ci, cr

def pairwise_editor(names, key):
    n=len(names)
    df=pd.DataFrame(1.0,index=names,columns=names)
    for i in range(n):
        for j in range(i+1,n):
            c1,c2=st.columns([1,1])
            with c1:
                v=st.number_input(f"{names[i]} dibanding {names[j]}", min_value=0.111, max_value=9.0, value=1.0, step=0.1, key=f"{key}_{i}_{j}")
            df.iloc[i,j]=v
            df.iloc[j,i]=1/v
    return df.values

st.title("📊 Sistem Pendukung Keputusan AHP")
st.caption("Studi Kasus: Pemilihan Vendor Kendaraan Operasional PT Pertamina Pertagas Niaga")
st.info("Catatan: nilai perbandingan pada aplikasi ini adalah contoh/simulasi. Ganti dengan data kuesioner penelitian Anda untuk hasil penelitian resmi.")

criteria=["Quality","Delivery","Cost","Service","Information Technology"]
vendors=["PT PMS","Vendor B","Vendor C","Vendor D"]

tab1,tab2,tab3=st.tabs(["1. Kriteria","2. Alternatif","3. Hasil"])

with tab1:
    st.header("Perbandingan Berpasangan Kriteria")
    cm=pairwise_editor(criteria,"criteria")
    cw,cn=priority(cm)
    lam,ci,cr=consistency(cm,cw)
    st.subheader("Matriks")
    st.dataframe(pd.DataFrame(cm,index=criteria,columns=criteria).round(3),use_container_width=True)
    st.subheader("Bobot Kriteria")
    st.dataframe(pd.DataFrame({"Kriteria":criteria,"Bobot":cw,"Persentase":cw*100}).sort_values("Bobot",ascending=False).round(4),use_container_width=True)
    a,b,c=st.columns(3)
    a.metric("λ Max",f"{lam:.4f}")
    b.metric("CI",f"{ci:.4f}")
    c.metric("CR",f"{cr:.4f}")
    st.success("KONSISTEN (CR < 0,10)" if cr < .10 else "TIDAK KONSISTEN (CR ≥ 0,10)")

with tab2:
    st.header("Perbandingan Alternatif Vendor")
    all_weights={}
    for criterion in criteria:
        st.subheader(f"Alternatif berdasarkan {criterion}")
        am=pairwise_editor(vendors,criterion.replace(" ","_"))
        w,_=priority(am)
        all_weights[criterion]=w
        lam,ci,cr=consistency(am,w)
        st.dataframe(pd.DataFrame({"Vendor":vendors,"Bobot":w,"Persentase":w*100}).sort_values("Bobot",ascending=False).round(4),use_container_width=True)
        st.caption(f"CR = {cr:.4f} — " + ("Konsisten" if cr < .10 else "Tidak konsisten"))

with tab3:
    st.header("Hasil Akhir AHP")
    if "cw" not in locals() or not all_weights:
        st.warning("Isi perbandingan pada tab sebelumnya.")
    else:
        aw=np.column_stack([all_weights[c] for c in criteria])
        final=aw@cw
        result=pd.DataFrame({"Vendor":vendors,"Nilai Akhir":final,"Persentase":final*100})
        result=result.sort_values("Nilai Akhir",ascending=False).reset_index(drop=True)
        result["Ranking"]=range(1,len(result)+1)
        st.dataframe(result.round(4),use_container_width=True)
        st.bar_chart(result.set_index("Vendor")["Nilai Akhir"])
        st.success(f"Vendor dengan prioritas tertinggi: {result.iloc[0]['Vendor']} ({result.iloc[0]['Nilai Akhir']:.4f})")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Prediksi Konsumsi Listrik Jepang", layout="wide")
st.title("Prediksi Konsumsi Listrik di Jepang")
st.write("Masukkan variabel ekonomi untuk memprediksi konsumsi listrik sektor Residential & Industrial.")

# --- Load dataset untuk ambil Region & mapping ---
dataset = pd.read_csv("electricity_data.csv")
region_list = dataset['Region'].unique().tolist()
region_list.sort()
region_mapping = {name: idx for idx, name in enumerate(region_list)}

# Load pipeline (Scaler + Model)
pipeline_res = joblib.load("pipeline_res.pkl")
pipeline_ind = joblib.load("pipeline_ind.pkl")

# --- Input user ---
region = st.selectbox("Region", options=region_list)
year = st.slider("Tahun", min_value=1990, max_value=2030, value=2025, step=1)

# Tabs untuk sektor
tab1, tab2 = st.tabs(["🏠 Residential", "🏭 Industrial"])
active_tab = None

with tab1:
    st.subheader("Input Variabel Residential")
    intensity_res = st.number_input("Electricity Intensity (kWh)", min_value=0.0, step=0.1, format="%.2f", key="intensity_res")
    nominal_price_res = st.number_input("Nominal Price (Yen/kWh)", min_value=0.0, step=0.01, format="%.2f", key="nominal_price_res")
    real_price_res = st.number_input("Real Price (Yen/kWh)", min_value=0.0, step=0.01, format="%.2f", key="real_price_res")
    if st.button("Prediksi Residential", key="btn_res"):
        active_tab = "res"

with tab2:
    st.subheader("Input Variabel Industrial")
    intensity_ind = st.number_input("Electricity Intensity (kWh)", min_value=0.0, step=0.1, format="%.2f", key="intensity_ind")
    nominal_price_ind = st.number_input("Nominal Price (Yen/kWh)", min_value=0.0, step=0.01, format="%.2f", key="nominal_price_ind")
    real_price_ind = st.number_input("Real Price (Yen/kWh)", min_value=0.0, step=0.01, format="%.2f", key="real_price_ind")
    if st.button("Prediksi Industrial", key="btn_ind"):
        active_tab = "ind"

# Format angka ribuan
def format_idr(num):
    return '{:,.2f}'.format(num).replace(',', 'X').replace('.', ',').replace('X', '.')

# --- Fungsi elastisitas sederhana ---
def estimate_elasticity(model, base_input, price_col, delta=0.05):
    higher_price = base_input.copy()
    higher_price[price_col] *= (1 + delta)
    base_pred = model.predict(base_input)[0]
    new_pred = model.predict(higher_price)[0]
    elasticity = ((new_pred - base_pred) / base_pred) / (delta)
    return elasticity

# --- Jalankan prediksi sesuai tab aktif ---
if active_tab in ["res", "ind"]:
    region_code = region_mapping[region]
    if active_tab == "res":
        input_data = pd.DataFrame([[region_code, year, intensity_res, nominal_price_res, real_price_res]],
                                  columns=["Region", "Year", "Intensity", "NominalPrice", "RealPrice"])
        model = pipeline_res
        label = "Residential"
    else:
        input_data = pd.DataFrame([[region_code, year, intensity_ind, nominal_price_ind, real_price_ind]],
                                  columns=["Region", "Year", "Intensity", "NominalPrice", "RealPrice"])
        model = pipeline_ind
        label = "Industrial"

    pred_value = model.predict(input_data)[0]

    st.subheader(f"Hasil Prediksi - {label}")
    st.write(f"**Prediksi Konsumsi:** {format_idr(pred_value)} kWh")

    # --- Accordion untuk tujuan penelitian ---
    with st.expander("🎯 Tujuan 1: Elastisitas & Tren Konsumsi"):
        years = list(range(1990, 2031))
        pred_list = []
        for y in years:
            temp = input_data.copy()
            temp["Year"] = y
            pred_list.append(model.predict(temp)[0])
        fig, ax = plt.subplots()
        ax.plot(years, pred_list, label=label, color="blue" if active_tab == "res" else "orange")
        ax.set_title(f"Tren Prediksi Konsumsi Listrik ({label}) - {region}")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Konsumsi (kWh)")
        ax.legend()
        st.pyplot(fig)

        elasticity = estimate_elasticity(model, input_data, "NominalPrice")
        st.write(f"**Estimasi Elastisitas Harga:** {elasticity:.3f} (interpretasi: {'inelastis' if abs(elasticity)<1 else 'elastis'})")

    with st.expander("🎯 Tujuan 2: Perbandingan Antar Region"):
        compare_preds = []
        for r_name, r_code in region_mapping.items():
            temp = input_data.copy()
            temp["Region"] = r_code
            compare_preds.append((r_name, model.predict(temp)[0]))
        compare_df = pd.DataFrame(compare_preds, columns=["Region", "Prediksi"])
        compare_df = compare_df.sort_values("Prediksi", ascending=False)
        fig2, ax2 = plt.subplots()
        ax2.bar(compare_df["Region"], compare_df["Prediksi"], color="green")
        ax2.set_title(f"Perbandingan Konsumsi Listrik per Region ({label})")
        ax2.set_ylabel("Prediksi Konsumsi (kWh)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with st.expander("🎯 Tujuan 3: Insight & Rekomendasi"):
        if pred_value > np.median(compare_df["Prediksi"]):
            insight = f"Konsumsi {label.lower()} di {region} relatif tinggi dibandingkan region lain. Perlu strategi pengendalian konsumsi dan efisiensi energi."
        else:
            insight = f"Konsumsi {label.lower()} di {region} relatif rendah. Masih ada ruang untuk optimalisasi pemanfaatan listrik."
        st.write("**Insight:**")
        st.write(insight)
        st.write("**Rekomendasi:**")
        st.write("- Evaluasi kebijakan harga listrik untuk menjaga keseimbangan konsumsi.")
        st.write("- Implementasi program efisiensi energi berbasis sektor.")
        st.write("- Prioritaskan wilayah dengan konsumsi tinggi untuk intervensi kebijakan.")

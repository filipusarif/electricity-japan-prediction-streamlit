import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Konsumsi Listrik Jepang", layout="wide")

@st.cache_resource
def load_models():
    pipeline_res = joblib.load("./models/pipeline_res.pkl")
    pipeline_ind = joblib.load("./models/pipeline_ind.pkl")
    return pipeline_res, pipeline_ind

pipeline_res, pipeline_ind = load_models()

region_map = {
    'Chubu': 0, 'Chugoku': 1, 'Hokkaido': 2, 'Hokuriku': 3, 'Kansai': 4,
    'Kyusyu': 5, 'Okinawa': 6, 'Shikoku': 7, 'Tohoku': 8, 'Tokyo': 9
}
region_names = list(region_map.keys())

st.title("Prediksi Konsumsi Listrik Jepang")
st.write("Masukkan variabel ekonomi untuk memprediksi konsumsi listrik sektor **Residential** dan **Industrial**.")

tab1, tab2 = st.tabs(["Residential", "Industrial"])

# ======================== RESIDENTIAL ==========================
with tab1:
    st.subheader("Input Data - Sektor Residential")

    col1, col2 = st.columns(2)
    with col1:
        region_res_name = st.selectbox("Region", options=region_names, key="region_res")
        year_res = st.number_input("Year", min_value=1990, max_value=2050, value=2025, key="year_res")
        intensity_res = st.number_input("Intensity", min_value=0.0, value=3000.0, key="intensity_res")
    with col2:
        nominal_price_res = st.number_input("Nominal Price", min_value=0.0, value=20.0, key="nominal_res")
        real_price_res = st.number_input("Real Price", min_value=0.0, value=150.0, key="real_res")

    region_res = region_map[region_res_name]

    input_res_df = pd.DataFrame([{
        "Region": region_res,
        "Year": year_res,
        "Intensity": intensity_res,
        "NominalPrice": nominal_price_res,
        "RealPrice": real_price_res
    }])

    if st.button("Prediksi Residential"):
        res_pred = pipeline_res.predict(input_res_df)[0]
        st.success(f"Prediksi Konsumsi Listrik Residential: {res_pred:,.2f} Kwh")

        # --- Summary Table ---
        st.write("#### Summary Input & Prediksi")
        summary_res = input_res_df.copy()
        summary_res["Prediksi_Konsumsi_KWh"] = round(res_pred, 2)
        summary_res["Region"] = region_res_name
        st.dataframe(summary_res)

        # --- Simulasi Konsumsi 10 Tahun ke Depan ---
        st.write("#### Simulasi Konsumsi 10 Tahun ke Depan (Residential)")
        future_years = list(range(year_res, year_res + 10))
        simulasi_df = pd.DataFrame({
            "Region": region_res,
            "Year": future_years,
            "Intensity": intensity_res,
            "NominalPrice": nominal_price_res,
            "RealPrice": real_price_res
        })
        simulasi_pred = pipeline_res.predict(simulasi_df)

        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(future_years, simulasi_pred, marker='o')
        ax1.set_xlabel("Tahun")
        ax1.set_ylabel("Konsumsi (KWh)")
        ax1.set_title("Prediksi 10 Tahun Mendatang")
        st.pyplot(fig1)

        # --- Simulasi Harga Riil ---
        st.write("####  Sensitivitas Harga Riil (Residential)")
        harga_range = list(range(int(real_price_res) - 50, int(real_price_res) + 60, 10))
        harga_df = pd.DataFrame({
            "Region": region_res,
            "Year": year_res,
            "Intensity": intensity_res,
            "NominalPrice": nominal_price_res,
            "RealPrice": harga_range
        })
        harga_preds = pipeline_res.predict(harga_df)

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(harga_range, harga_preds, marker='s', color='green')
        ax2.set_xlabel("Real Price")
        ax2.set_ylabel("Konsumsi (KWh)")
        ax2.set_title("Sensitivitas terhadap Harga")
        st.pyplot(fig2)

# ======================== INDUSTRIAL ==========================
with tab2:
    st.subheader("Input Data - Sektor Industrial")

    col3, col4 = st.columns(2)
    with col3:
        region_ind_name = st.selectbox("Region", options=region_names, key="region_ind")
        year_ind = st.number_input("Year", min_value=1990, max_value=2050, value=2025, key="year_ind")
        intensity_ind = st.number_input("Intensity", min_value=0.0, value=50000.0, key="intensity_ind")
    with col4:
        nominal_price_ind = st.number_input("Nominal Price", min_value=0.0, value=30.0, key="nominal_ind")
        real_price_ind = st.number_input("Real Price", min_value=0.0, value=140.0, key="real_ind")

    region_ind = region_map[region_ind_name]

    input_ind_df = pd.DataFrame([{
        "Region": region_ind,
        "Year": year_ind,
        "Intensity": intensity_ind,
        "NominalPrice": nominal_price_ind,
        "RealPrice": real_price_ind
    }])

    if st.button("Prediksi Industrial"):
        ind_pred = pipeline_ind.predict(input_ind_df)[0]
        st.success(f"Prediksi Konsumsi Listrik Industrial: {ind_pred:,.2f} Kwh")

        # --- Summary Table ---
        st.write("#### Summary Input & Prediksi")
        summary_ind = input_ind_df.copy()
        summary_ind["Prediksi_Konsumsi_KWh"] = round(ind_pred, 2)
        summary_ind["Region"] = region_ind_name
        st.dataframe(summary_ind)

        # --- Simulasi Konsumsi 10 Tahun ke Depan ---
        st.write("#### Simulasi Konsumsi 10 Tahun ke Depan (Industrial)")
        future_years_ind = list(range(year_ind, year_ind + 10))
        simulasi_ind_df = pd.DataFrame({
            "Region": region_ind,
            "Year": future_years_ind,
            "Intensity": intensity_ind,
            "NominalPrice": nominal_price_ind,
            "RealPrice": real_price_ind
        })
        simulasi_ind_pred = pipeline_ind.predict(simulasi_ind_df)

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(future_years_ind, simulasi_ind_pred, marker='o')
        ax3.set_xlabel("Tahun")
        ax3.set_ylabel("Konsumsi (KWh)")
        ax3.set_title("Prediksi 10 Tahun Mendatang")
        st.pyplot(fig3)

        # --- Simulasi Harga Riil ---
        st.write("####  Sensitivitas Harga Riil (Industrial)")
        harga_range_ind = list(range(int(real_price_ind) - 50, int(real_price_ind) + 60, 10))
        harga_ind_df = pd.DataFrame({
            "Region": region_ind,
            "Year": year_ind,
            "Intensity": intensity_ind,
            "NominalPrice": nominal_price_ind,
            "RealPrice": harga_range_ind
        })
        harga_preds_ind = pipeline_ind.predict(harga_ind_df)

        fig4, ax4 = plt.subplots(figsize=(10, 3))
        ax4.plot(harga_range_ind, harga_preds_ind, marker='s', color='orange')
        ax4.set_xlabel("Real Price")
        ax4.set_ylabel("Konsumsi (KWh)")
        ax4.set_title("Sensitivitas terhadap Harga")
        st.pyplot(fig4)

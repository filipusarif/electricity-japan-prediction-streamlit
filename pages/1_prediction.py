import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Konsumsi Listrik Jepang", layout="wide")

@st.cache_resource
def load_models():
    pipeline_res = joblib.load("models/pipeline_res.pkl")
    pipeline_ind = joblib.load("models/pipeline_ind.pkl")
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
        
        # Buat kesimpulan tren otomatis
        start_year, end_year = future_years[0], future_years[-1]
        start_val, end_val = simulasi_pred[0], simulasi_pred[-1]

        if end_val > start_val:
            trend = "peningkatan"
        elif end_val < start_val:
            trend = "penurunan"
        else:
            trend = "stabil"

        kesimpulan = (
            f"Prediksi konsumsi listrik sektor Residential untuk {len(future_years)} tahun mendatang "
            f"menunjukkan {trend} dari sekitar {start_val/1e9:.1f} miliar kWh pada {start_year} "
            f"menjadi {end_val/1e9:.1f} miliar kWh pada {end_year}. "
        )

        st.write("##### Penjelasan Hasil Konsumsi Listrik")
        st.write(kesimpulan)

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
        
        # --- Kesimpulan otomatis sensitivitas harga ---
        start_price, end_price = harga_range[0], harga_range[-1]
        start_cons, end_cons = harga_preds[0], harga_preds[-1]

        if end_cons > start_cons:
            arah = "berbanding lurus"
        elif end_cons < start_cons:
            arah = "berbanding terbalik"
        else:
            arah = "tidak menunjukkan perubahan signifikan"

        persen_perubahan = ((end_cons - start_cons) / start_cons) * 100

        kesimpulan_harga = (
            f"Konsumsi listrik sektor Residential cenderung {arah} terhadap harga riil. "
            f"Ketika harga naik dari {start_price} menjadi {end_price}, konsumsi berubah dari "
            f"{start_cons/1e9:.2f} miliar kWh menjadi {end_cons/1e9:.2f} miliar kWh "
            f"({persen_perubahan:.2f}% perubahan)."
        )

        st.write("##### Penjelasan Hasil Sensitivitas Harga")
        st.write(kesimpulan_harga)

        # === Kesimpulan Gabungan untuk Kebijakan ===
        final_kesimpulan = (
            f"Analisis menunjukkan bahwa konsumsi listrik sektor Residential {arah} terhadap harga riil. "
            f"Kenaikan harga dari {start_price} menjadi {end_price} berpotensi mengubah konsumsi "
            f"dari {start_cons/1e9:.2f} miliar kWh menjadi {end_cons/1e9:.2f} miliar kWh "
            f"({persen_perubahan:+.2f}%). "
            f"Selain itu, prediksi jangka {len(future_years)} tahun ke depan memperlihatkan {trend} konsumsi "
            f"dari {start_val/1e9:.1f} miliar kWh pada {start_year} menjadi {end_val/1e9:.1f} miliar kWh pada {end_year}. "
            f"Temuan ini memberikan dasar penting bagi pembuat kebijakan untuk menyesuaikan strategi harga listrik "
            f"dan merancang program efisiensi energi yang mampu mengendalikan permintaan "
            f"serta menjaga keberlanjutan pasokan."
        )

        st.write("### Kesimpulan untuk Kebijakan")
        st.write(final_kesimpulan)

        

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
        
        # Buat kesimpulan tren otomatis
        start_year, end_year = future_years_ind[0], future_years_ind[-1]
        start_val, end_val = simulasi_ind_pred[0], simulasi_ind_pred[-1]

        if end_val > start_val:
            trend = "peningkatan"
        elif end_val < start_val:
            trend = "penurunan"
        else:
            trend = "stabil"

        kesimpulan = (
            f"Prediksi konsumsi listrik sektor Industrial untuk {len(future_years_ind)} tahun mendatang "
            f"menunjukkan {trend} dari sekitar {start_val/1e9:.1f} miliar kWh pada {start_year} "
            f"menjadi {end_val/1e9:.1f} miliar kWh pada {end_year}. "
        )

        st.write("##### Penjelasan Hasil Konsumsi Listrik")
        st.write(kesimpulan)

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
        
        # --- Kesimpulan otomatis sensitivitas harga ---
        start_price, end_price = harga_range_ind[0], harga_range_ind[-1]
        start_cons, end_cons = harga_preds_ind[0], harga_preds_ind[-1]

        if end_cons > start_cons:
            arah = "berbanding lurus"
        elif end_cons < start_cons:
            arah = "berbanding terbalik"
        else:
            arah = "tidak menunjukkan perubahan signifikan"

        persen_perubahan = ((end_cons - start_cons) / start_cons) * 100

        kesimpulan_harga = (
            f"Konsumsi listrik sektor Residential cenderung {arah} terhadap harga riil. "
            f"Ketika harga naik dari {start_price} menjadi {end_price}, konsumsi berubah dari "
            f"{start_cons/1e9:.2f} miliar kWh menjadi {end_cons/1e9:.2f} miliar kWh "
            f"({persen_perubahan:.2f}% perubahan)."
        )

        st.write("##### Penjelasan Hasil Sensitivitas Harga")
        st.write(kesimpulan_harga)
        
        # === Kesimpulan Gabungan untuk Kebijakan ===
        final_kesimpulan_ind = (
            f"Analisis menunjukkan bahwa konsumsi listrik sektor Industrial {arah} terhadap harga riil. "
            f"Kenaikan harga dari {start_price} menjadi {end_price} berpotensi mengubah konsumsi "
            f"dari {start_cons/1e9:.2f} miliar kWh menjadi {end_cons/1e9:.2f} miliar kWh "
            f"({persen_perubahan:+.2f}%). "
            f"Selain itu, prediksi jangka {len(future_years_ind)} tahun ke depan memperlihatkan {trend} konsumsi "
            f"dari {start_val/1e9:.1f} miliar kWh pada {start_year} menjadi {end_val/1e9:.1f} miliar kWh pada {end_year}. "
            f"Temuan ini memberikan dasar penting bagi pembuat kebijakan untuk menyesuaikan strategi harga listrik "
            f"dan merancang program efisiensi energi yang mampu mengendalikan permintaan "
            f"serta menjaga keberlanjutan pasokan."
        )

        st.write("### Kesimpulan untuk Kebijakan")
        st.write(final_kesimpulan_ind)

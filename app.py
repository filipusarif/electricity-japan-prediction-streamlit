import streamlit as st
import xgboost as xgb
import pandas as pd

st.title("Prediksi Konsumsi Listrik di Jepang")
st.write("Masukkan variabel ekonomi untuk memprediksi konsumsi listrik sektor Residential & Industrial.")

# Load model XGBoost
res_model = xgb.Booster()
res_model.load_model("xgboost_residential.json")

ind_model = xgb.Booster()
ind_model.load_model("xgboost_industrial.json")

# Input user
price = st.number_input("Harga Listrik (Yen/kWh)", min_value=0.0, step=0.1)
gdp = st.number_input("GDP (Triliun Yen)", min_value=0.0, step=0.1)
temp = st.number_input("Temperatur (°C)", min_value=-10.0, max_value=40.0, step=0.1)

if st.button("Prediksi"):
    # Buat DataFrame
    input_data = pd.DataFrame([[price, gdp, temp]], columns=["price", "gdp", "temperature"])
    dmatrix = xgb.DMatrix(input_data)
    
    # Prediksi
    res_pred = res_model.predict(dmatrix)[0]
    ind_pred = ind_model.predict(dmatrix)[0]
    
    st.subheader("Hasil Prediksi")
    st.write(f"**Sektor Residential:** {res_pred:.2f} GWh")
    st.write(f"**Sektor Industrial:** {ind_pred:.2f} GWh")


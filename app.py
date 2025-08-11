import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Home - Konsumsi Listrik Jepang",
    page_icon="⚡",
    layout="wide"
)

st.title("Analisis Konsumsi Listrik Jepang")
st.write("informasi dan visualisasi dari dataset konsumsi listrik Jepang.")

df = pd.read_csv("./datasets/electricity_data.csv")

st.subheader("Dataset")
st.dataframe(df.head())

# ====== Visualisasi 1: Time Series ======
st.subheader("Konsumsi Listrik dari Tahun ke Tahun")
region_selected = st.selectbox("Pilih Region", df["Region"].unique())
df_region = df[df["Region"] == region_selected]

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_region["Year"], df_region["Con. Res"], marker='o', label="Residential")
ax1.plot(df_region["Year"], df_region["Con. Ind"], marker='s', label="Industrial")
ax1.set_xlabel("Tahun")
ax1.set_ylabel("Konsumsi (KWh)")
ax1.set_title(f"Trend Konsumsi Listrik - {region_selected}")
ax1.legend()
st.pyplot(fig1)

# ====== Visualisasi 2: Hubungan Harga Nominal vs Konsumsi ======
st.subheader("Hubungan Harga Nominal dan Konsumsi")
sector_choice = st.radio("Pilih Sektor", ["Residential", "Industrial"])

if sector_choice == "Residential":
    price_col = "N. Price. Res"
    consumption_col = "Con. Res"
else:
    price_col = "N. Price. Ind"
    consumption_col = "Con. Ind"

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df, x=price_col, y=consumption_col, hue="Region", ax=ax2)
ax2.set_title(f"Harga Nominal vs Konsumsi - {sector_choice}")
st.pyplot(fig2)

# ====== Visualisasi 3: Perbandingan Konsumsi per Region ======
st.subheader("Perbandingan Konsumsi Listrik per Region")
avg_consumption = df.groupby("Region")[["Con. Res", "Con. Ind"]].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(8, 5))
avg_consumption.plot(kind="bar", x="Region", ax=ax3)
ax3.set_ylabel("Rata-rata Konsumsi (KWh)")
ax3.set_title("Rata-rata Konsumsi Residential vs Industrial per Region")
st.pyplot(fig3)

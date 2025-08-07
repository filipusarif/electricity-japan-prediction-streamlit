# 🔌 Electricity Consumption Prediction in Japan (1990–2015)

A **Streamlit web app** for predicting electricity consumption in **Japan’s Residential and Industrial sectors**, using economic indicators and time series data from **1990 to 2015**.

👉 **[Try the App Online](https://electricity-japan-prediction.streamlit.app/#input-data-sektor-residential)**

---

## 📊 Features

- 🔄 **Dual-sector prediction**: Residential & Industrial
- ✍️ **Interactive input**: Enter economic variables like intensity and price
- 📈 **Visualization**: Historical vs. predicted consumption plots
- 🧠 **Machine learning models**: Ridge & ElasticNet regression

---

## 🧾 Input Variables

The app allows users to enter the following variables for each sector:

### 🔹 Residential Sector

| Variable        | Description                                   | Example      |
|-----------------|-----------------------------------------------|--------------|
| Region          | Geographic area in Japan                      | `Chubu`      |
| Year            | Prediction year (must be ≥ 2015)              | `2025`       |
| Intensity       | Energy consumption intensity (kWh/m²)         | `3000.00`    |
| Nominal Price   | Electricity price before inflation adjustment | `20.00`      |
| Real Price      | Adjusted price accounting for inflation       | `18.50`      |

### 🔸 Industrial Sector

Same structure as Residential, with separate input form.

---

## 🧠 Models Used

Regression models trained using 1990–2015 time series data:

- **Ridge Regression**
- **ElasticNet Regression**

Trained separately for Residential and Industrial sectors.

---

## 🖼️ Output

- ✅ Predicted electricity consumption values
- 📊 Time series plots showing historical vs predicted trends
- 📋 Tabular summary of predictions

---

## 🚀 Run Locally

### 1. Clone the repository (if applicable)

```bash
git clone https://github.com/filipusarif/electricity-japan-prediction-streamlit.git
cd electricity-japan-prediction-streamlit
```

### 2. Install dependencies
Make sure Python 3.7+ is installed, then install all required packages using:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

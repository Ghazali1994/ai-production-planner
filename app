import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

st.title("🧮 Production Planner AI Tool")

# Upload historical data
uploaded_file = st.file_uploader("Upload past orders data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview:", df.head())

    if 'Order Quantity' in df.columns and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayNumber'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['DayNumber']]
        y = df['Order Quantity']

        model = LinearRegression()
        model.fit(X, y)

        st.success("✅ Model trained on historical data.")

        # Forecast
        days_to_forecast = st.slider("Days to forecast", 1, 30, 7)
        future_days = pd.DataFrame({'DayNumber': [df['DayNumber'].max() + i for i in range(1, days_to_forecast + 1)]})
        predictions = model.predict(future_days)

        forecast_df = pd.DataFrame({
            "Forecast Date": [df['Date'].max() + datetime.timedelta(days=i) for i in range(1, days_to_forecast + 1)],
            "Predicted Order Quantity": predictions.astype(int)
        })

        st.write("📈 Forecast:", forecast_df)

        csv = forecast_df.to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')

    else:
        st.error("Uploaded file must contain 'Order Quantity' and 'Date' columns.")

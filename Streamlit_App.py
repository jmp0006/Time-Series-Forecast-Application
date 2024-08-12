import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def load_csv():
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    return df_input

def prep_data(df, date_col, metric_col):
    df_input = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
    st.success("The selected date column is now labeled as **ds** and the Target column as **y**")
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    return df_input

# TITLE & DESCRIPTION
st.title('New Claim Forecast with Prophet')
st.write('This app enables you to generate a time series forecast for new claims.')
st.write("""
    ### How to Use the App

    1. **Upload Dataset**: Begin by uploading your dataset using the file uploader. Ensure that your dataset contains a date column and a target column to be forecasted.

    2. **Select Columns**: Choose the appropriate columns for date and target from the dropdown menus.

    3. **Adjust Forecast Parameters**: Set the number of months to forecast and adjust the changepoint and seasonality scales to tune the model.

    4. **View Forecast**: Once the model is fitted, view the forecast plot, including the forecasted values, trend, and seasonality components.

    5. **Explore Additional Plots**: Check out the components of the forecast and model performance metrics for more insights.
""")

df = pd.DataFrame()

# DATA LOADING
st.subheader('Data Loading')

# UPLOADING DATASET
Data = st.radio('Data', ['Upload Dataset'], index=0, horizontal=True, label_visibility='collapsed', key='Data')

if Data == 'Upload Dataset':
    input = st.file_uploader('Upload a Dataset')
    if input is not None:
        with st.spinner('Loading data..'):
            df = load_csv()

if not df.empty:
    columns = list(df.columns)
    
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select date column", index=0, options=columns, key="date", help='Column to be parsed as a date')
    with col2:
        metric_col = st.selectbox("Select Target column", index=1, options=columns, key="values", help='Quantity to be forecasted')
    
    df = prep_data(df, date_col, metric_col)
    output = 0
    
    Options = st.radio('Options', ['Plot Time-Series Data', 'Show Dataframe'], horizontal=True, label_visibility='collapsed', key='options')

    if Options == 'Plot Time-Series Data':
        st.line_chart(df.set_index('ds'), use_container_width=True, height=300)

    if Options == 'Show Dataframe':
        st.dataframe(df, use_container_width=True)

    # Forecasting with Prophet
    st.subheader('Forecasting with Prophet')

    # User input for forecast duration
    months_to_forecast = st.slider('Select number of months to forecast', min_value=1, max_value=48, value=12)
    
    # User input for changepoint and seasonality scales
    st.write("### Changepoint and Seasonality Scales")
    st.write("Adjusting the **changepoint scale** affects the model's sensitivity to changes in the trend. A higher value allows the model to fit more flexible trends, while a lower value results in a more rigid trend.")
    changepoint_scale = st.slider('Changepoint Scale', min_value=0.001, max_value=0.5, value=0.05, step=0.001)
    st.write("Adjusting the **seasonality scale** influences the model's sensitivity to seasonal effects. A higher value allows the model to fit more pronounced seasonal patterns, while a lower value makes the model less sensitive to seasonality.")
    seasonality_scale = st.slider('Seasonality Scale', min_value=0.001, max_value=10.0, value=1.0, step=0.001)
    
    # Fit the Prophet model with user-defined scales
    model = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_prior_scale=seasonality_scale)
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=months_to_forecast, freq='M')
    forecast = model.predict(future)

    # Plot forecast
    st.write("### Forecasted Values with Confidence Intervals")

    fig1 = model.plot(forecast, xlabel='Date', ylabel='Forecasted Values')
    plt.title('Forecasted vs Actual Data')
    st.pyplot(fig1)

    # Display forecast table
    st.write("### Forecast Data Table")
    st.write("""
    This table displays the forecasted values, along with their upper and lower bounds, for the selected number of months into the future.
    """)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months_to_forecast)
    forecast_df = forecast_df.rename(columns={'yhat': 'Forecast', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
    # Round values to 0 decimal places
    forecast_df = forecast_df.round(0)
    # Remove timestamp from ds column
    forecast_df['ds'] = forecast_df['ds'].dt.date
    st.dataframe(forecast_df, use_container_width=True)
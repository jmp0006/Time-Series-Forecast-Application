import streamlit as st
import pandas as pd
import numpy as np

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.serialize import model_to_json, model_from_json
import holidays

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

import itertools
from datetime import datetime
    
def load_csv():  
    df_input = pd.DataFrame()    
    df_input=pd.read_csv(input,sep=None, engine='python', encoding='utf-8',
                            parse_dates=True,
                            infer_datetime_format=True)
    return df_input

def prep_data(df, date_col, metric_col):
    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.success("The selected date column is now labeled as **ds** and the Target column as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input

# TITLE & DESCRIPTION
st.title('Time-Series Forecasting with Prophet')
st.write('This app enables you to generate time series forecast.')

with st.sidebar:
    image_url = "https://raw.githubusercontent.com/MuskaanMehra/Time-Series-Forecast-Application/main/Assets/Streamlit_image.png"
    st.image(image_url, use_column_width=True)
    st.header('_**Streamlit Prophet**_')
    st.subheader('A web-app to train, evaluate and optimise forecasting model in a visual way using **[Prophet](https://facebook.github.io/prophet/)**.')

df =  pd.DataFrame()   

#SELECTING DATASET-- DATA LOADING
st.subheader('Data loading')

#UPLOADING DATASET
input = st.file_uploader('Upload a Dataset')

if input is not None:
    with st.spinner('Loading data..'):
        df = load_csv()

if not df.empty:
        columns = list(df.columns)
        
        col1,col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select date column",index= 0,options=columns,key="date", help='Column to be parsed as a date')
        with col2:
            metric_col = st.selectbox("Select Target column",index=1,options=columns,key="values", help='Quantity to be forecasted')

        df = prep_data(df, date_col, metric_col)
        output = 0

        Options = st.radio('Options', ['Plot Time-Seies data', 'Show Dataframe', 'Show Descriptive Statistics'], horizontal=True, label_visibility='collapsed', key = 'options')
        
        if Options == 'Plot Time-Seies data':
            try:
                line_chart = alt.Chart(df).mark_line().encode(
                    x = 'ds:T',
                    y = "y:Q",tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
                st.altair_chart(line_chart,use_container_width=True)
                
            except:
                st.line_chart(df['y'],use_container_width =True,height = 300)

        if Options =='Show Dataframe':
            st.dataframe(df, use_container_width=True)
        if Options == 'Show Descriptive Statistics':
            st.write(df.describe().T, use_container_width=True)

# PARAMETERS CONFIGURATION

st.subheader("Parameters configuration")
with st.container():
    st.write('In this section you can modify the algorithm settings.')
            
    with st.expander("**Horizon**"):
        periods_input = st.number_input('Select how many future periods (days) to forecast.',
        min_value = 1, max_value = 366,value=90)

    with st.expander("**Seasonality**"):
        st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required.""")
        seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

    with st.expander("**Trend components**"):
        st.write("Add or remove components:")
        daily = st.checkbox("Daily")
        weekly= st.checkbox("Weekly")
        monthly = st.checkbox("Monthly")
        yearly = st.checkbox("Yearly")

    with st.expander("**Growth model**"):
        st.write('Prophet uses by default a linear growth model.')         

        growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

        if growth == 'linear':
            growth_settings= {
                            'cap':1,
                            'floor':0
                        }
            cap=1
            floor=1
            df['cap']=1
            df['floor']=0

        if growth == 'logistic':
            st.info('Configure saturation')

            cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
            floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
            if floor > cap:
                st.error('Invalid settings. Cap must be higher than floor.')
                growth_settings={}

            if floor == cap:
                st.warning('Cap must be higher than floor')
            else:
                growth_settings = {
                    'cap':cap,
                    'floor':floor
                    }
                df['cap']=cap
                df['floor']=floor

    with st.expander('**Holidays**'):    
        countries = ['Country name','United States','India', 'United Kingdom', 'France','Germany']
        with st.container():
            years=[2023]
            selected_country = st.selectbox(label="Select country",options=countries)

            if selected_country == 'India':
                for date, name in sorted(holidays.IN(years=years).items()):
                    st.write(date,name) 
                            
            if selected_country == 'United Kingdom':
                for date, name in sorted(holidays.GB(years=years).items()):
                    st.write(date,name)                      

            if selected_country == 'United States':                   
                for date, name in sorted(holidays.US(years=years).items()):
                    st.write(date,name)

            if selected_country == 'France':                    
                for date, name in sorted(holidays.FR(years=years).items()):
                    st.write(date,name)
                            
            if selected_country == 'Germany':                    
                for date, name in sorted(holidays.DE(years=years).items()):
                    st.write(date,name)

            else:
                holidays = False
                            
            holidays = st.checkbox('Add country holidays to the model')

    with st.expander('**Hyperparameters**'):
        st.write('In this section it is possible to tune the scaling coefficients.')
            
        seasonality_scale_values= [0.1, 1.0,5.0,10.0]    
        changepoint_scale_values= [0.01, 0.1, 0.5,1.0]

        st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
        changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            
        st.write("The seasonality change point controls the flexibility of the seasonality.")
        seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)

#FORECAST
with st.container():
    st.subheader("Forecast")
    st.write("Fit the model on the data and generate future prediction.")

    if input:            
        if st.checkbox("Initialize model (Fit)",key="fit"):
            if len(growth_settings)==2:
                m = Prophet(seasonality_mode=seasonality,
                            daily_seasonality=daily,
                            weekly_seasonality=weekly,
                            yearly_seasonality=yearly,
                            growth=growth,
                            changepoint_prior_scale=changepoint_scale,
                            seasonality_prior_scale= seasonality_scale)
                if holidays:
                    m.add_country_holidays(country_name=selected_country)
                        
                if monthly:
                    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

                with st.spinner('Fitting the model..'):
                    m = m.fit(df)
                    future = m.make_future_dataframe(periods=periods_input,freq='D')
                    future['cap']=cap
                    future['floor']=floor
                    st.write("The model will produce forecast up to ", future['ds'].max())
                    st.success('Model fitted successfully')

            else:
                st.warning('Invalid configuration')
           
        if st.checkbox("Generate forecast (Predict)",key="predict"):
            try:
                with st.spinner("Forecasting.."):
                    forecast = m.predict(future)
                    st.success('Prediction generated successfully')
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)
                    fig1 = m.plot(forecast)
                    st.write('Forecast plot')
                    st.write(fig1)                   
                    #st.pyplot(fig1, use_container_width=True)
                    output = 1

                if growth == 'linear':
                    fig2 = m.plot(forecast)
                    a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                    st.write(fig2)
                    output = 1
            except:
                st.warning("You need to train the model first.. ")

        if st.checkbox('Show components'):
            try:
                with st.spinner("Loading.."):
                    fig3 = m.plot_components(forecast)
                    st.write(fig3)
            except: 
                st.warning("Requires forecast generation..")
   
# CROSS-VALIDATION    
    st.subheader('Model validation')
    st.write("In this section it is possible to do cross-validation of the model.")
    with st.expander("**Explanation**"):
        st.markdown("""The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
        st.write("Training data (initial): The amount of data set aside for training.")
        st.write("Horizon: The data set aside for validation.")
        st.write("Period: a forecast is made for every observed point between cutoff and cutoff + horizon.""")

    with st.expander("**Cross-validation**"):    
        initial = st.number_input(value= 180,label="initial",min_value=30)

        period = st.number_input(value= 30,label="period",min_value=30,max_value=365)

        horizon = st.number_input(value= 30, label="horizon",min_value=30,max_value=365)

    with st.expander("**Metrics**"):            
        if input:
            if output == 1:
                metrics = 0
            if st.checkbox('Calculate metrics'):
                 with st.spinner("Cross validating.."):
                    try:
                        df_cv = cross_validation(m, initial=int(initial),
                                                period=int(period), 
                                                horizon = int(horizon),
                                                parallel="processes")                                                                  
                            
                        df_p= performance_metrics(df_cv)
                        # PERFORMANCE METRICS TABLE
                        st.dataframe(df_p)
                        metrics = 1

                    except Exception as e:
                        st.error(f"Error during Cross-validation: {e}")
                        metrics=0

                    if metrics == 1:
                        metrics = ['mse','rmse','mae','mape','mdape','coverage']
                        selected_metric = st.selectbox("Select metric to plot",options=metrics)
                        if selected_metric:
                            fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                            st.write(fig4)

        else:
            st.write("Create a forecast to see metrics")

# HYPER-PARAMETER TUNING
    st.subheader('Hyperparameter Tuning')
    st.write("In this section it is possible to find the best combination of hyperparamenters.")

    param_grid = {'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
                  'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],}

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    if input:
        if output == 1:

            if st.button("Optimize hyperparameters"):            
                with st.spinner("Finding best combination. Please wait.."):                        
                    try:
                    # Use cross validation to evaluate all parameters
                        for params in all_params:
                            m = Prophet(**params).fit(df)  # Fit model with given params
                            df_cv = cross_validation(m, initial=initial,
                                                            period=period,
                                                            horizon=horizon,
                                                            parallel="processes")
                            df_p = performance_metrics(df_cv, rolling_window=1)
                            rmses.append(df_p['rmse'].values[0])
                    except Exception as e:
                        st.error(f"Error during Hyperparameter optimization: {e}")
                        

                # Find the best parameters
                tuning_results = pd.DataFrame(all_params)
                tuning_results['rmse'] = rmses
                st.dataframe(tuning_results)
                            
                best_params = all_params[np.argmin(rmses)]
                    
                st.write('The best parameter combination is:')
                st.write(best_params)
                
        else:
            st.write("Create a model to optimize") 

    st.subheader('Export results')     
    st.write("Finally you can export your result forecast.")
        
    if input:
        if output == 1:               
            forecast_df = pd.DataFrame(forecast[['ds','yhat','yhat_lower','yhat_upper']])
            forecast_data = forecast_df.to_csv(index=False)
            st.download_button(label="Download Forecast CSV",
                                data=forecast_data,
                                file_name=f'Forecast_results.csv',
                                mime='text/csv',
                                key='download forecast')
            st.dataframe(forecast_df, use_container_width=True)
             
         
                
   
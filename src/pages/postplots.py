import streamlit as st
# import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Predicted Sales visualisation  📈 📊')

        data = pd.read_csv('src/pages/sub_plot.csv', index_col = 2)
        # st.sidebar.title("Predicted Sales Seasonality")
        # st.sidebar.subheader("Choose Feature or Aspect to plot")
        # plot = st.sidebar.selectbox("feature", ("Seasonality", "Open", 'Promotions', 'State Holiday', 'Assortment', 'Store Type','Competition'))
        # if st.sidebar.button("View Data", key='Display'):
        st.sidebar.title('Predicted data')
        st.sidebar.subheader('Input date ranges')
        # data = data.set_index('Date', inplace=True)
        start_date = st.sidebar.date_input('start date', datetime.date(2015,8,1))
        end_date = st.sidebar.date_input('end date', datetime.date(2015,9,20))
        # mask = (data['Date'] > start_date) & (data['Date'] <= end_date)
        # dates = data.index[mask]
        # date_mask = (data.index > start) & (data.index < end)
        # dis = data.loc[dates]
        # dis = data.loc[start_date:end_date]
        dis = data[data.index.isin(pd.date_range(start_date, end_date))]
        st.write(dis)
            
        # if st.sidebar.button("Predict", key='predict'):
        st.subheader("Weekly Averaged Predicted Sales Seasonality Plot")
        time_data = data[['Sales']]
        time_data['datetime'] = pd.to_datetime(time_data.index)
        time_data = time_data.set_index('datetime')
        # time_data = time_data.drop(['Date'], axis = 1)
        monthly_time_data = time_data.Sales.resample('D').mean() 
        plt.figure(figsize = (15,7))
        plt.title('Seasonality plot averaged weekly')
        plt.ylabel('average predicted sales')
        monthly_time_data.plot()
        plt.grid()
        st.pyplot()
        st.write("""
        The trends across the months cannot be observed given the predictions is 2 months long.
        Nevertheless, the plot is informative enough. 
        The trend observed captures the low sales during Sundays (2nd, 9th, 16th, 23rd, 30th August and 6th, 13th September.)
        From the train data, it is observed that most stores are closed on Sundays, hence the predicted sales for Sundays.
        The sales peak on Mondays then flatten during the remaining 5 days of the week.
        """)
            



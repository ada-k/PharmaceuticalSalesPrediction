''' This script takes care of displaying the predicted data and the corresponding plot.'''

# libraries
import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Predicted Sales and its visualisation  📈 📊')

        # read the data
        data = pd.read_csv('src/pages/sub_plot.csv', index_col = 2)

        st.sidebar.title("Predicted Sales Seasonality")
        st.sidebar.subheader('Input date ranges')

        # make the index a datetime object
        data.Date = data.index
        data.Date = pd.to_datetime(data.Date)

        # create inputs for date ranges
        start_date = st.sidebar.text_input('start date', "2015-9-19")
        end_date = st.sidebar.text_input('end date', "2015-9-20")
        # convert the inputs to timestamps
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        # filter the data using the inputs
        date_slice = np.logical_and(data.Date >= start_date, data.Date <= end_date)
        # notice that if you are using more than 2 comparisons use logical_and.reduce([test1, test2, test2])
        sliced_df = data[date_slice]

        # create an input for store id
        st.sidebar.subheader('Input Store ID')
        store_id = st.sidebar.number_input('Store ID', 1)
        store_data = sliced_df.loc[sliced_df.Store == store_id]

        # write the final data (filter by date and store)
        st.write(store_data)


        # plot the predicted data
        st.subheader("Weekly Averaged Predicted Sales Seasonality Plot")
        time_data = data[['Sales']]
        time_data['datetime'] = pd.to_datetime(time_data.index)
        time_data = time_data.set_index('datetime')
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

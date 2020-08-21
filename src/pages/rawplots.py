import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Raw Data Visualisation  📈 📊')

        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('/home/ada/stuff/streamlit 101/sales/train.csv', na_values=na_value)
        store = pd.read_csv('/home/ada/stuff/streamlit 101/sales/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        st.sidebar.title("Gallery")
        st.sidebar.subheader("Choose Feature or Aspect to plot")
        plot = st.sidebar.selectbox("feature", ("Seasonality", "Open", 'Promotions', 'State Holiday', 'Assortment', 'Store Type','Competition'))


        if plot == 'Seasonality':
            
            # if st.sidebar.button("Predict", key='predict'):
            st.subheader("Monthly Averaged Sales Seasonality Plot")
            time_data = full_train[['Date', 'Sales']]
            time_data['datetime'] = pd.to_datetime(time_data['Date'])
            time_data = time_data.set_index('datetime')
            time_data = time_data.drop(['Date'], axis = 1)
            monthly_time_data = time_data.Sales.resample('M').mean() 
            plt.figure(figsize = (15,7))
            plt.title('Seasonality plot averaged monthly')
            plt.ylabel('average sales')
            monthly_time_data.plot()
            plt.grid()
            st.pyplot()
            st.write("""
            Across the 3 years, there’s a shoot in the sales in the month of December.
            The peak in December can be explained by the Christmas holiday. A holidays implies numerous interactions and activities. 
            The sales take a sudden drop immediately after December (in January) and after April (in May). 
            This can be explained by the end of the Christmas and Easter holidays and getting back to ‘business as usual.’ 
            The impact of the other holidays: Easter and Public holidays are not as visible as that of Christmas. 
            This is because they take a shorter duration, thus the cumulative effect cannot be well established.
            """)
            
        if plot == 'Open':
            st.subheader("Open status in relation to day of the week")
            fig, (axis1) = plt.subplots(1,1,figsize=(16,8))
            sns.countplot(x='Open',hue='DayOfWeek', data=full_train, palette="husl", ax=axis1)
            plt.title("store's open status in relation to day of the week")
            st.pyplot()
            st.write("""
            Most of the stores are open in the first 6 days and closed on the 7th. Implying Sundays are their only rest days.
            """)


        if plot == 'Promotions':
            st.subheader("Countplot and Barplots indicating Promotions and Sales and customers across the stores")
            sns.countplot(x='Promo', data=full_train).set_title('Promo counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='Promo', y='Sales', data=full_train, ax=axis1).set_title('sales across different Promo')
            sns.barplot(x='Promo', y='Customers', data=full_train, ax=axis2).set_title('customers across different Promo')
            st.pyplot()
            st.write("""
            Less stores run the promotions.
            This could be as a result of a bunch of reasons, one being extra costs incurred. 
            Despite the number of stores running promos on a daily basis being less, their sales are almost twice that of the stores running no promo.
            The promos prove useful in increasing the volume of sales, thus stores that can’t afford to run a promo on a daily basis should subscribe to the continuous consecutive plans.
            """)

        if plot == 'State Holiday':
            st.subheader("Sales During State Holidays and Ordinary Days")
            full_train["StateHoliday"].loc[full_train["StateHoliday"] == 0] = "0"
            # value counts
            sns.countplot(x='StateHoliday', data=full_train).set_title('State holidays value counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))
            # full_train["StateHoliday"] = full_train["StateHoliday"].loc[full_train["StateHoliday"] == 0] = "0"
            sns.barplot(x='StateHoliday', y='Sales', data=full_train, ax=axis1).set_title('comparison of sales during StateHolidays and ordinary days')
            # holidays only      
            mask = (full_train["StateHoliday"] != "0") & (full_train["Sales"] > 0)
            sns.barplot(x='StateHoliday', y='Sales', data=full_train[mask], ax=axis2).set_title('sales during Stateholidays')
            st.pyplot()
            st.write("""
            The sales are less during the holidays since most of the stores are closed on holidays 
            and also because the number of holidays are less compared to ordinary days.
            **a** is representative of public holidays, **b** - Easter and **c** -Christmas.
            The sales are higher during Christmas and Easter holidays.
            """)

        # if plot == 'School Holiday':
        #     st.subheader("Sales During School Holidays and Ordinary Days")
        #     sns.countplot(x='SchoolHoliday', data=full_train).set_title('a count plot of school holidays')
        #     st.pyplot()
        #     fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

        #     sns.barplot(x='SchoolHoliday', y='Sales', data=full_train, ax=axis1).set_title('sales across ordinary school days and school holidays')
        #     sns.barplot(x='SchoolHoliday', y='Customers', data=full_train, ax=axis2).set_title('no of customers across ordinary school days and school holidays')
        #     st.pyplot()
        #     st.write("""
        #     The sales are less during the holidays since most of the stores are closed on holidays 
        #     and also because the number of holidays are less compared to ordinary days.
        #     a is representative of public holidays, b - Easter and c -Christmas.
        #     The sales are higher during Christmas and Easter holidays.
        #     """)

        if plot == 'Assortment':
            st.subheader("Sales across different assortment types")
            sns.countplot(x='Assortment', data=full_train, order=['a','b','c']).set_title('assortment types counts')
            st.pyplot()
            # fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='Assortment', y='Sales', data=full_train, order=['a','b','c']).set_title('sales across different assortment types')
            # sns.barplot(x='Assortment', y='Customers', data=full_train, order=['a','b','c'], ax=axis2).set_title('sales across different assortment types')

            st.pyplot()
            st.write("""
            The store counts in the 3 assortment classes. Basic(a) and extended(c) are the most populated.
            The sales volumes across the 3 classes. Despite  the extra(b) class having the least number of stores, it has the highest volume of sales.
            """)

        if plot == 'Store Type':
            st.subheader("Sales across different store types")
            sns.countplot(x='StoreType', data=full_train, order=['a','b','c', 'd']).set_title('a count plot of StoreTypes')
            st.pyplot()
            # fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='StoreType', y='Sales', data=full_train, order=['a','b','c', 'd']).set_title('sales across different StoreType')
            # sns.barplot(x='StoreType', y='Customers', data=full_train, ax=axis2, order=['a','b','c', 'd']).set_title('no of customers across diffrent StoreType')
            st.pyplot()
            st.write("""
            Type a is the most popular store type, while b is the least popular.
            Despite b being the least popular, it records the highest amount of sales.
            """)

        if plot == 'Competition':
            st.subheader("Sales across different competition distance decile groups")
            # adding Decile_rank column to the DataFrame 
            full_train['Decile_rank'] = pd.qcut(full_train['CompetitionDistance'], 5, labels = False) 
            new_df = full_train[['Decile_rank', 'Sales']]
            # a = new_df.groupby('Decile_rank').sum()
            a = new_df.groupby('Decile_rank').mean()

            #plot the decile classes
            plt.figure(figsize=(8,6))
            sns.barplot(x = a.index, y = a.Sales)
            plt.title('Total sales per decile group')
            plt.ylabel('sales')
            plt.xlabel('decile rank')
            st.pyplot()
            st.write("""
            The length of competition distances increase with decile classes. The total number of sales across the decile classes is 
            somewhat balanced, apart from the first class which has a bit higher values compared to the rest. We expect it to have a
            lower volume  considering the competition aspect but another 
            argument that could explain the opposite behavior is the stores location.
            They could be located in big cities where population is dense thus  proximity to competitive stores has a minor influence.

            """)
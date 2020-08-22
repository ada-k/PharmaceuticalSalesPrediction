''' This scripts takes care of explatory data analysis. It deals mostlt with visualisations of the raw data.'''

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

        # read the datasets
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        st.sidebar.title("Gallery")
        st.sidebar.subheader("Choose Feature or Aspect to plot")
        plot = st.sidebar.selectbox("feature", ("Seasonality", "Correlation", "SchoolHoliday", "Open/DayOfWeek", 'Promotions', 'State Holiday', 'PromoIntervals', 'Assortment', 'Store Type','Competition'))

        # SchoolHoliday plots
        if plot == 'SchoolHoliday':
            st.subheader("School Holidays")
            sns.countplot(x='SchoolHoliday', data=full_train, palette = 'Set2').set_title('a count plot of school holidays')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='SchoolHoliday', y='Sales', data=full_train, ax=axis1, palette = 'Set2').set_title('sales across ordinary school days and school holidays')
            sns.barplot(x='SchoolHoliday', y='Customers', data=full_train, ax=axis2, palette = 'Set2').set_title('no of customers across ordinary school days and school holidays')
            st.pyplot()
            # st.write("""
            # Not so many stores were affected by the closure of schools.
            # But for the few affected, their sales don't 
            # """)

        # Competition plots
        if plot == 'Competition':
            st.subheader("Competition Distance")
            # adding Decile_rank column to the DataFrame 
            full_train['Decile_rank'] = pd.qcut(full_train['CompetitionDistance'], 5, labels = False) 
            new_df = full_train[['Decile_rank', 'Sales']]
            # a = new_df.groupby('Decile_rank').sum()
            a = new_df.groupby('Decile_rank').mean()
            labels = a.index.to_list()
            sizes = a.Sales.to_list()
            fig = plt.figure(figsize =(10, 7)) 
            colors = ['gold', 'yellowgreen', 'purple', 'lightcoral', 'lightskyblue']
            explode = (0.1, 0.03, 0.03, 0.03, 0.03)  # explode 1st slice

            # Plot
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%.2f', startangle=140)
            plt.title('A piechart indicating mean sales in the 5 CompetitioDIstance decile classes')
            st.pyplot()

            # adding Decile_rank column to the DataFrame 
            full_train['Decile_rank'] = pd.qcut(full_train['CompetitionDistance'], 5, labels = False) 
            new_df = full_train[['Decile_rank', 'Customers']]
            # a = new_df.groupby('Decile_rank').sum()
            a = new_df.groupby('Decile_rank').mean()
            labels = a.index.to_list()
            sizes = a.Customers.to_list()
            fig = plt.figure(figsize =(10, 7)) 
            colors = ['gold', 'yellowgreen', 'purple', 'lightcoral', 'lightskyblue']
            explode = (0.1, 0.03, 0.03, 0.03, 0.03)  # explode 1st slice

            # Plot
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%.2f', startangle=140)
            plt.title('A piechart indicating mean number of customers in the 5 CompetitioDistance decile classes')
            st.pyplot()
            st.write("""
            The length of competition distances increase with decile classes. The total number of sales across the decile classes is 
            somewhat balanced, apart from the first class which has a bit higher values compared to the rest. We expect it to have a
            lower volume  considering the competition aspect but another 
            argument that could explain the opposite behavior is the stores location.
            They could be located in big cities where population is dense thus  proximity to competitive stores has a minor influence.
            """)


        # Seasonality plots
        if plot == 'Seasonality':
            
            # if st.sidebar.button("Predict", key='predict'):
            st.subheader("Daily, Weekly and Monthly Averaged Sales Seasonality Plot")

            time_data = full_train[['Date', 'Sales']]
            time_data['datetime'] = pd.to_datetime(time_data['Date'])
            time_data = time_data.set_index('datetime')
            time_data = time_data.drop(['Date'], axis = 1)

            # daily train
            daily_time_data = time_data.Sales.resample('D').mean() 
            plt.figure(figsize = (12,5))
            plt.figure(figsize = (12,5))
            plt.title('Seasonality plot averaged daily')
            daily_time_data.plot()
            plt.grid() 
            st.pyplot()  

            # weekly train
            weekly_time_data = time_data.Sales.resample('W').mean() 
            plt.figure(figsize = (12,5))
            plt.title('Seasonality plot averaged weekly')
            plt.ylabel('average sales')
            weekly_time_data.plot()
            plt.grid()
            st.pyplot()

            #monthly
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

        # Correlation plots
        if plot == 'Correlation':            
            # if st.sidebar.button("Predict", key='predict'):
            st.subheader("Linear Relationships between the Sales and the predictor features")
            def correlation_map(f_data, f_feature, f_number):
                f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
                f_correlation = f_data[f_most_correlated].corr()
                
                f_mask = np.zeros_like(f_correlation)
                f_mask[np.triu_indices_from(f_mask)] = True
                with sns.axes_style("white"):
                    f_fig, f_ax = plt.subplots(figsize=(8, 6))
                    f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                                    annot=True, annot_kws={"size": 10}, cmap="BuPu")

                plt.show()

            print('top 6 features with highest correlation with sales')
            correlation_map(full_train, 'Sales', 6)
            st.pyplot()
            st.write("""
            Sales and Customers have a high correlation. 
            This is because sales are directly dependent on the number of customers.
            """)


        # Open/DayOfWeek plots
        if plot == 'Open/DayOfWeek':
            st.subheader("Open status in relation to day of the week")
            fig, (axis1) = plt.subplots(1,1,figsize=(16,8))
            sns.countplot(x='Open',hue='DayOfWeek', data=full_train, palette="RdBu_r", ax=axis1)
            plt.title("store's open status in relation to day of the week")
            st.pyplot()
            # sales across dayofweek
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='DayOfWeek', y='Sales', data=full_train, palette = 'RdBu_r', ax=axis1).set_title('sales across different days of the week ')
            sns.barplot(x='DayOfWeek', y='Customers', data=full_train, palette = 'RdBu_r', ax=axis2).set_title('customers across different days of the week ')
            st.pyplot()
            st.write("""
            Most of the stores are open in the first 6 days and closed on the 7th. Implying Sundays are their only rest days.
            """)


        # PromoIntervals plots
        if plot == 'PromoIntervals':
            st.subheader("Promotion Intervals")
            flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            sns.countplot(x='PromoInterval', data=full_train, palette = flatui).set_title('PromoInterval value counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='PromoInterval', y='Sales', data=full_train, ax=axis1, palette = flatui).set_title('sales across different promo intervals')
            sns.barplot(x='PromoInterval', y='Customers', data=full_train, ax=axis2, palette = flatui).set_title('customers across different promo intervals')
            st.pyplot()
            # st.write("""
            # Most of the stores are open in the first 6 days and closed on the 7th. Implying Sundays are their only rest days.
            # """)


        # Promotions plots
        if plot == 'Promotions':
            flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            st.subheader("Countplot and Barplots indicating Promotions and Sales and customers across the stores")
            sns.countplot(x='Promo', data=full_train, palette = flatui).set_title('Promo counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='Promo', y='Sales', data=full_train, ax=axis1, palette = flatui).set_title('sales across different Promo')
            sns.barplot(x='Promo', y='Customers', data=full_train, ax=axis2, palette = flatui).set_title('customers across different Promo')
            st.pyplot()
            st.write("""
            Less stores run the promotions.
            This could be as a result of a bunch of reasons, one being extra costs incurred. 
            Despite the number of stores running promos on a daily basis being less, their sales are almost twice that of the stores running no promo.
            The promos prove useful in increasing the volume of sales, thus stores that can’t afford to run a promo on a daily basis should subscribe to the continuous consecutive plans.
            """)


        # State Holiday plots
        if plot == 'State Holiday':
            st.subheader("Sales During State Holidays and Ordinary Days")
            full_train["StateHoliday"].loc[full_train["StateHoliday"] == 0] = "0"
            # value counts
            sns.countplot(x='StateHoliday', data=full_train, palette = 'Paired').set_title('State holidays value counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))
            # full_train["StateHoliday"] = full_train["StateHoliday"].loc[full_train["StateHoliday"] == 0] = "0"
            sns.barplot(x='StateHoliday', y='Sales', data=full_train, ax=axis1, palette = 'Paired').set_title('comparison of sales during StateHolidays and ordinary days')
            # holidays only      
            mask = (full_train["StateHoliday"] != "0") & (full_train["Sales"] > 0)
            sns.barplot(x='StateHoliday', y='Sales', data=full_train[mask], ax=axis2, palette = 'Paired').set_title('sales during Stateholidays')
            st.pyplot()

            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))
            sns.barplot(x='StateHoliday', y='Customers', data=full_train, ax=axis1, palette = 'Paired').set_title('comparison of customers during StateHolidays and ordinary days')
            # holidays only
            mask = (full_train["StateHoliday"] != "0") & (full_train["Customers"] > 0)
            sns.barplot(x='StateHoliday', y='Customers', data=full_train[mask], ax=axis2, palette = 'Paired').set_title('customers during Stateholidays')
            st.pyplot()
            st.write("""
            The sales are less during the holidays since most of the stores are closed on holidays 
            and also because the number of holidays are less compared to ordinary days.
            **a** is representative of public holidays, **b** - Easter and **c** -Christmas.
            The sales are higher during Christmas and Easter holidays.
            """)



        # Assortment plots
        if plot == 'Assortment':
            st.subheader("Sales across different assortment types")
            sns.countplot(x='Assortment', data=full_train, order=['a','b','c'], palette = 'husl').set_title('assortment types counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='Assortment', y='Sales', data=full_train, order=['a','b','c'], palette = 'husl', ax = axis1).set_title('sales across different assortment types')
            sns.barplot(x='Assortment', y='Customers', data=full_train, order=['a','b','c'], ax=axis2, palette = 'husl').set_title('Number of customers across different assortment types')
            st.pyplot()
            st.write("""
            The store counts in the 3 assortment classes. Basic(a) and extended(c) are the most populated.
            The sales volumes across the 3 classes. Despite  the extra(b) class having the least number of stores, it has the highest volume of sales.
            """)

        # Store Type plots
        if plot == 'Store Type':
            st.subheader("Sales across different store types")
            sns.countplot(x='StoreType', data=full_train, order=['a','b','c', 'd'], palette = ["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]).set_title('a count plot of StoreTypes')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='StoreType', y='Sales', data=full_train, ax = axis1, order=['a','b','c', 'd'], palette = ["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]).set_title('sales across different StoreType')
            sns.barplot(x='StoreType', y='Customers', data=full_train, ax=axis2, order=['a','b','c', 'd'], palette = ["#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]).set_title('no of customers across diffrent StoreType')
            st.pyplot()
            st.write("""
            Type a is the most popular store type, while b is the least popular.
            Despite b being the least popular, it records the highest amount of sales.
            """)

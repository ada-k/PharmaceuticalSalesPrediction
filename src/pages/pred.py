# libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from scipy import stats
from scipy.stats import skew, norm

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import datetime

import warnings
warnings.filterwarnings(action="ignore")

        
        
        
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Data ..."):
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('Sales Predictions 💰 🛍️ 💳 💸')
        st.write("""
        Predictions and the accuracy of the predictions.
        """)
  
    
    @st.cache()
    def load_preprocess_data():

        # load data
        global train_features, test_features, train_target, full_test, full_train, train, test, store, submission, categorical, numerical
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        test = pd.read_csv('src/pages/test.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        submission = pd.read_csv('src/pages/sample_submission.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_test = pd.merge(left = test, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')  

        # '''preprocessing'''
        
        # train and target features
        train_features = full_train.drop(['Sales', 'Customers'], axis = 1) #drop the target feature + customers (~ will not be used for prediction)
        train_target  = full_train[['Sales']]
        test_features = full_test.drop(['Id'], axis = 1) #drop id, it's required only during submission
        
        #feature generation + transformations
        train_features['Date'] = pd.to_datetime(train_features.Date)
        train_features['Month'] = train_features.Date.dt.month.to_list()
        train_features['Year'] = train_features.Date.dt.year.to_list()
        train_features['Day'] = train_features.Date.dt.day.to_list()
        train_features['WeekOfYear'] = train_features.Date.dt.weekofyear.to_list()
        train_features['DayOfWeek'] = train_features.Date.dt.dayofweek.to_list()
        train_features['weekday'] = 1        # Initialize the column with default value of 1
        train_features.loc[train_features['DayOfWeek'] == 5, 'weekday'] = 0
        train_features.loc[train_features['DayOfWeek'] == 6, 'weekday'] = 0
        # train_features = train_features.drop(['Date'], axis = 1)
        train_features = train_features.drop(['Store'], axis = 1)

        test_features['Date'] = pd.to_datetime(test_features.Date)
        test_features['Month'] = test_features.Date.dt.month.to_list()
        test_features['Year'] = test_features.Date.dt.year.to_list()
        test_features['Day'] = test_features.Date.dt.day.to_list()
        test_features['WeekOfYear'] = test_features.Date.dt.weekofyear.to_list()
        test_features['DayOfWeek'] = test_features.Date.dt.dayofweek.to_list()
        test_features['weekday'] = 1        # Initialize the column with default value of 1
        test_features.loc[test_features['DayOfWeek'] == 5, 'weekday'] = 0
        test_features.loc[test_features['DayOfWeek'] == 6, 'weekday'] = 0
        # test_features = test_features.drop(['Date'], axis = 1)
        test_features = test_features.drop(['Store'], axis = 1)
        
        
        # numerical and categorical columns (train set)
        categorical = []
        numerical = []
        timestamp = []

        for col in train_features.columns:
            if train_features[col].dtype == object:
                categorical.append(col)
            elif train_features[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                numerical.append(col)
            else:
                timestamp.append(col)

        # Keep selected columns only
        my_cols = categorical + numerical + timestamp
        train_features = train_features[my_cols].copy()
        test_features = test_features[my_cols].copy()
        features = pd.concat([train_features, test_features]) #merge the features columns for uniform preprocessing

        # change dtypes for uniformity in preprocessing
        features.CompetitionOpenSinceMonth = features.CompetitionOpenSinceMonth.astype('Int64') 
        features.CompetitionOpenSinceYear = features.CompetitionOpenSinceYear.astype('Int64')
        features.Promo2SinceWeek = features.Promo2SinceWeek.astype('Int64') 
        features.Promo2SinceYear = features.Promo2SinceYear.astype('Int64')
        features["StateHoliday"].loc[features["StateHoliday"] == 0] = "0"
        
        
        
        # ''' actual preprocessing: the mighty pipeline '''
        # numeric
        for col in ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']:
            features[col] = features[col].fillna((int(features[col].mean()))) 
        features.PromoInterval = features.PromoInterval.fillna(features.PromoInterval.mode()[0])
        features.Open = features.Open.fillna(features.Open.mode()[0])
        features = pd.get_dummies(features, columns=['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday'], drop_first=True)


        # the mighty pipeline
        # categorical_transformer = Pipeline(steps=[
        #     ('imputer', SimpleImputer(strategy='most_frequent', missing_values = np.nan))
        # ])

        # categorical_encoder = Pipeline(steps = [
        #     ('encoder', OneHotEncoder())
        # ])

        # numerical_transformer = Pipeline(steps=[
        #     ('imputer', SimpleImputer(strategy='mean', missing_values = np.nan)),
        #     # ('scaler', RobustScaler())
        # ])
        # numerical_scaler = Pipeline(steps = [
        #     ('scaler', RobustScaler())
        # ])

        # # Bundle preprocessing for numerical and categorical data
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         # ('num', numerical_transformer, ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']),
        #         # ('cat', categorical_transformer, ['Open']),
        #         # ('cat_map', categorical_encoder, ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']),
        #         ('scaler', numerical_scaler, numerical)
        #     ])

        # my_pipeline = Pipeline(steps=[('preprocessor', preprocessor) ])
        # transformed_features = my_pipeline.fit_transform(features)
        # features = pd.DataFrame(transformed_features)
        scaler = RobustScaler()
        c = ['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'WeekOfYear', 'Month', 'Year', 'Day', 'WeekOfYear', 'weekday']
        features[numerical] = scaler.fit_transform(features[numerical].values)
        


        
        # # categorical
        # indices = []
        # for col in ['Open', 'PromoInterval']:
        #     k = features.columns.get_loc(col)
        #     indices.append(k)

        # #imputing with column mode.
        # columns = indices
        # for col in columns:
        #     x = features.iloc[:, col].values
        #     x = x.reshape(-1,1)
        #     imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        #     imputer = imputer.fit(x)
        #     x = imputer.transform(x)
        #     features.iloc[:, col] = x
            
            
        # # numeric
        # # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # # columns = []
        # for col in ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']:
        #     features[col] = features[col].fillna((int(features[col].mean())))
        

        # for col in columns:
        #     x = features.iloc[:, col].values
        #     x = x.reshape(-1,1)
        #     imputer = imputer.fit(x)
        #     x = imputer.transform(x)
        #     features.iloc[:, col] = x
                
                
        # one hot encoder (categorical variables)
        # features = pd.get_dummies(features, columns=['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday'], drop_first=True)
        
        return features
    
    
    # @st.cache(persist=True)
    @st.cache()
    def reconstruct_sets(features):
        global x_train, x_val, y_train, y_val
        # global train_set
        # original train and test sets
        x_train = features.iloc[:len(train_features), :]
        x_test = features.iloc[len(train_features):, :]
        y_train = train_target
        # train_set = pd.concat([x_train, y_train], axis=1)
        
        # updated train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .20, random_state = 0)

        
        return x_train, x_val, y_train, y_val, x_test
    
    
    
    features = load_preprocess_data()
    features = features.drop(['Date'], axis = 1)

    x_train, x_val, y_train, y_val, x_test = reconstruct_sets(features)
    # log transformation on target variable
    y_train = np.log1p(y_train['Sales'])
    y_val = np.log1p(y_val['Sales'])

    st.sidebar.title("Predictions")
    st.sidebar.subheader("Choose Model")
    regressor = st.sidebar.selectbox("Regressor", ("Random Forest Regressor", "eXtreme Gradient Boosting(XGB)", "Gradient Boosting"))
    
    
    def display_metrics(metrics_list):
        if 'Mean Absolute Error' in metrics_list:
            st.subheader("Mean Absolute Error")
            print(mean_absolute_error(y_pred, y_val))
            st.write('Mean absolute erro:', mean_absolute_error(y_pred, y_val))

        if 'Mean Squared Error' in metrics_list:
            st.subheader("Mean Squared Error")
            print(mean_squared_error(y_pred, y_val))
            st.write('Mean squared error:', mean_squared_error(y_pred, y_val))

    # global y_pred
    # random forest
    if regressor == 'Random Forest Regressor':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        estimators = st.sidebar.number_input("n_estimators", 100, 400, step=10, key='n_estimators')
        max_features = st.sidebar.radio("max_features", ("auto", "sqrt", "log2"), key='max_features')

        metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            st.subheader("Random Forest Regressor")
            model = RandomForestRegressor(n_estimators=estimators, max_features=max_features, random_state = 42)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            st.write("Mean Absolute Error: ", mean_absolute_error(y_val, y_pred).round(4))
            st.write("Mean Squared Error: ", mean_squared_error(y_val, y_pred).round(4))
            display_metrics(metrics)
            predictions = model.predict(x_test)

            # if st.sidebar.checkbox("Show predicted data", True):
            st.subheader("Rossmann Pharmaceuticals sales predictions")
            # size = st.sidebar.number_input("n_rows", 1, 2000, step=7, key='number of rows')
            sub = full_test[['Id']]
            back = np.expm1(predictions)
            sub['Sales'] = back
            # sub = sub.sort_values(by = 'Id', ascending = True)
            sub['Date'] = full_test.Date.to_list()
            sub.to_csv('sub.csv', index = False)
            sub['Store'] = full_test.Store.to_list()
            sub['Date'] = pd.to_datetime(sub['Date'])
            start_date = st.sidebar.date_input('start date', datetime.date(2015,8,1))
            end_date = st.sidebar.date_input('end date', datetime.date(2015,9,20))
            mask = (sub['Date'] > start_date) & (sub['Date'] <= end_date)
            dis = sub.loc[mask]
            st.write(dis)

    # xgb
    # global y_pred
    if regressor == 'eXtreme Gradient Boosting(XGB)r':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        eval_metric = st.sidebar.radio("eval_metric", ("rmse", "mae"), key='eval_metric')
        booster = st.sidebar.radio("booster", ("gbtree", "gblinear"), key='booster')

        metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            st.subheader("eXtreme Gradient Boosting(XGB)")
            model = XGBRegressor(eval_metric=eval_metric, booster=booster, random_state = 42)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            st.write("Mean Absolute Error: ", mean_absolute_error(y_val, y_pred).round(4))
            st.write("Mean Squared Error: ", mean_squared_error(y_val, y_pred).round(4))
            display_metrics(metrics)
            predictions = model.predict(x_test)

            # if st.sidebar.checkbox("Show predicted data", True):
            st.subheader("Rossmann Pharmaceuticals sales predictions")
            # size = st.sidebar.number_input("n_rows", 1, 2000, step=7, key='number of rows')
            sub = full_test[['Id']]
            back = np.expm1(predictions)
            sub['Sales'] = back
            # sub = sub.sort_values(by = 'Id', ascending = True)
            sub['Date'] = full_test.Date.to_list()
            sub['Store'] = full_test.Store.to_list()
            sub['Date'] = pd.to_datetime(sub['Date'])
            # sub.to_csv('sub.csv', index = False)
            sub['Store'] = full_test.Store.to_list()
            sub['Date'] = pd.to_datetime(sub['Date'])
            start_date = st.sidebar.date_input('start date', datetime.date(2015,8,1))
            end_date = st.sidebar.date_input('end date', datetime.date(2015,9,20))
            mask = (sub['Date'] > start_date) & (sub['Date'] <= end_date)
            dis = sub.loc[mask]
            st.write(dis)


    # grad boost
    # global y_pred
    if regressor == 'Gradient Boosting':
        st.sidebar.subheader("Model Hyperparameters")

        metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            st.subheader("Gradient Boosting")
            model = GradientBoostingRegressor(random_state = 42)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            st.write("Mean Absolute Error: ", mean_absolute_error(y_val, y_pred).round(4))
            st.write("Mean Squared Error: ", mean_squared_error(y_val, y_pred).round(4))
            display_metrics(metrics)
            predictions = model.predict(x_test)

            # if st.sidebar.checkbox("Show predicted data", True):
            st.subheader("Rossmann Pharmaceuticals sales predictions")
            # size = st.sidebar.number_input("n_rows", 1, 2000, step=7, key='number of rows')
            sub = full_test[['Id']]
            back = np.expm1(predictions)
            sub['Sales'] = back
            # sub = sub.sort_values(by = 'Id', ascending = True)
            sub['Date'] = full_test.Date.to_list()
            # sub.to_csv('sub.csv', index = False)
            sub['Store'] = full_test.Store.to_list()
            sub['Date'] = pd.to_datetime(sub['Date'])
            start_date = st.sidebar.date_input('start date', datetime.date(2015,8,1))
            end_date = st.sidebar.date_input('end date', datetime.date(2015,9,20))
            mask = (sub['Date'] > start_date) & (sub['Date'] <= end_date)
            dis = sub.loc[mask]
            st.write(dis)
  

    # # display raw data
    # if st.sidebar.checkbox("Show raw data", True):
    #     st.subheader("Rossmann Pharmaceuticals Data Set")
    #     st.write(features.head(10))
    #     st.markdown("The data and feature description for this challenge can be found [here.]()https://www.kaggle.com/c/rossmann-store-sales")
    
    
    # display predicted sales
    # if st.sidebar.checkbox("Show predictions data", True):
    #     st.subheader("Rossmann Pharmaceuticals sales predictions")
    #     size = st.sidebar.number_input("n_rows", 1, 2000, step=7, key='number of rows')
    #     sub = full_test[['Id']]
    #     back = np.expm1(predictions)
    #     sub['Sales'] = back
    #     sub = sub.sort_values(by = 'Id', ascending = True)
    #     # sub.to_csv('sub.csv', index = False)
    #     st.write(sub.head(size))

        
if __name__ == '__main__':
    main()


    

        
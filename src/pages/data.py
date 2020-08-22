import streamlit as st
import pandas as pd 

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Data ..."):
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('Data description  ⓿ ➀ ➁ ➂ 🔢 ')
        st.write("""
        Most of the data fields are easy to understand, but just to highlight some of the features present:
        **Store, Date, Sales, Customers, Open, State Holiday, School Holiday, Store Type, Assortment, Competition and Promotion.**

        The *Store Type, Assortment, Competition* and *Promotion* features are store tailored.

        The *Sales, Customers, Open, State Holiday* and *School Holiday* features vary across the stores with days.
        """)
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_train = full_train.set_index('Store')
        st.write(full_train.sample(20))

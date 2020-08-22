'''This writes the insights gained from the EDA.'''

import streamlit as st
import awesome_streamlit as ast

def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Data ..."):
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('INSIGHTS FROM THE DATA')
        st.image('src/pages/index.png', use_column_width=True)
        st.markdown("""
        The data has a lot of useful features that can provide introspections into the stores sales.
        Based on the explatory data analysis conducted, the following conclusions can be made: 
        * The number of customers is directly related to the volume of sales.
        * Store type **b** is the least popular while **a** is the most popular. The volume of sales are highest in store types**b**.
        * Assortment category **b** is the least popular while **a** is the most popular. The volume of sales are highest in category **b**.
        * Most stores are open from Monday to Saturday and closed on Sunday.The amount of sales and number of customers align with the trend across the week.
        * Not most stores run daily promotions. The sales volume and customers number are higher in the less stores that run daily promotions.
        * In the state holidays category, sales are highest during Christmas followed closely by Easter. The other holidays have a fairly low volume of sales.
        * Competition doesn't necessarily result to low sales volumes in competing stores. High competition is an indication of cities, thus 
        a fairly good number of customers are availble for both the competing stores.
        """)

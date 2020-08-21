import streamlit as st

import awesome_streamlit as ast
import src.pages.home
import src.pages.data 
import src.pages.rawplots
import src.pages.pred
import src.pages.postplots

ast.core.services.other.set_logging_format()

PAGES = {
    "Home": src.pages.home,
    "Raw Data":src.pages.data,
    "Raw Data visualisations": src.pages.rawplots,
    "Predictions": src.pages.pred,
    "Predictions visualisations": src.pages.postplots,
}


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This App is an end-to-end product that enables the Rosemann pharmaceutical company to 
        view predictions on sales across their stores and 6 weeks ahead of time and the trends expected.
"""
    )


if __name__ == "__main__":
    main()
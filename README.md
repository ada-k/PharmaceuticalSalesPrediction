# PharmaceuticalSalesPrediction

A challenge whose main task is to come up with an end to product that delivers Sales forecasting across multiples stores of Rossman Pharmaceutical company.

Main tasks:

  * **Regression**: Predicting sales across multiple stores for the coming 6 weeks. The performance of 4 regression models are explored: Linear Regression, XGBoost, Random Forest and GradBoost. Random Forest regressor emerges the best performer with a Mean Square Error of 0.01876.

  * **Time series analysis**:  Checking for trends and seasonality in the existing records and drawing conclusions based on the results.                                                                                                 

  * **Model Deployment**:  Integrating the regression models into existing production environment using streamlit, a python library for building web-apps.

  * **Hosting**: Making the web app accessible through heroku. 

Data used is from [kaggle](https://www.kaggle.com/c/rossmann-store-sales/data).
The [web-app](https://rossman-sales-pred.herokuapp.com/) is hosted with heroku. 

**Main pages of the app:**
* Home Pages
* Raw Data 
* Raw Data Visualisations
* Run predictions
* Predicted data + visualisations
* Insights.

### Project Structure
`kernels`contains the following notebooks:
* eda-rossmanstores.ipynb -eda notebook
* salespred.ipynb - ML notebook
`licence`License template
`readme`Markdown file giving a brief description of the project and its structure.
`Procfile`Heroku main dyno
`app.py`Streamlit's mother script
`src` Streamlit scripts/pages
`runtime.txt`Defining python version for heroku
`worker.py`Heroku worker dyno

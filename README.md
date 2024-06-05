# Retail-store-forecasting-model
Note: This README file provides an overview and instructions for running the Favorita Grocery Sales Forecasting project in a Kaggle notebook environment.

## Overview
This project involves forecasting grocery sales for the Corporación Favorita, a large Ecuadorian-based grocery retailer. The goal is to predict the unit sales for thousands of items sold at different Favorita stores across Ecuador. Accurate sales forecasting is crucial for managing inventory, optimizing supply chain operations, and improving customer satisfaction.

## Dataset
The dataset used in this project is sourced from the Kaggle competition "Corporación Favorita Grocery Sales Forecasting". It contains transactional data along with additional datasets providing information about stores, items, promotions, holidays, and more. The key datasets used include:

train.csv: Historical daily sales data
test.csv: Data for which sales predictions are required
items.csv: Metadata for items
stores.csv: Metadata for stores
oil.csv: Daily oil price data
holidays_events.csv: Information on holidays and events
transactions.csv: Daily transactions for each store

## Instructions

### Running the Project on Kaggle
Due to memory resource limitations, this project is designed to be executed in the Kaggle notebook environment. Follow these steps to run the notebooks:
Run favorita-grocery-store-case-study.ipynb notebook to execute both training and inference workflows.

### Results

NWRMSLE OF Training dataset = 0.78
NWRMSLE OF Val dataset 1 = 0.79
NWRMSLE OF Val dataset 2 = 0.78

## Conclusion
This project aims to provide an end-to-end solution for grocery sales forecasting using the Favorita dataset. By following the structured approach outlined in the notebooks, you will be able to preprocess the data, engineer features, train models, and generate sales forecasts for submission.


Note: This README file provides an overview and instructions for running the Favorita Grocery Sales Forecasting project in a Kaggle notebook environment. Adjustments may be required based on the specific configurations and limitations when running on local machines.

# Housing Price Prediction

## Overview

This project uses machine learning algorithms to predict housing prices based on various features like the number of bedrooms, square footage, and neighborhood. It demonstrates techniques in data preprocessing, model training, and evaluation.

## Dataset

The dataset used for this project is from Kaggle's House Prices Competition:
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Housing_Price_Prediction.git
    cd Housing_Price_Prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the dataset (`house_prices.csv`) in the `/data` folder.

4. Run the following scripts:
    - `data_preprocessing.py`: To preprocess and clean the data.
    - `model.py`: To train the machine learning model.
    - `evaluation.py`: To evaluate the model's performance.

## Usage

### Data Preprocessing
Run the following script to preprocess the data:

```bash
python scripts/data_preprocessing.py

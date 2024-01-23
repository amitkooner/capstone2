# New York Airbnb Open Data 2024 Project

## 1. Problem Description and Solution Usage

### Problem Statement
This project focuses on analyzing Airbnb data to build a predictive model for rental prices. The challenge lies in accurately predicting the price of a listing based on features such as location, room type, number of beds, and more. Accurate predictions can help hosts price their listings competitively and assist guests in finding the best value for their stays.

### Solution Overview
I developed a machine learning pipeline to predict Airbnb rental prices. The solution involves extensive data preprocessing, exploratory data analysis (EDA), and training several regression models to predict the listing prices. Models like Linear Regression, Ridge Regression, Lasso Regression, and ensemble methods were used for the predictions.

### How the Solution Will Be Used
The model can be used by Airbnb hosts to optimize the pricing of their listings and by guests to estimate the fair price of a rental. Additionally, it can be utilized by Airbnb internally to suggest optimal pricing strategies to new hosts or to flag listings with anomalous pricing.

The data used in this project was sourced from [New York Dataset on Kaggle](https://www.kaggle.com/datasets/vrindakallu/new-york-dataset).

## 2. Installation and Environment Setup

### Prerequisites
- Python 3.8+
- Pip package manager

### Setting Up a Virtual Environment

To isolate the project dependencies, it's recommended to use a virtual environment:

```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 3. Running the Project

To run the project, first activate the virtual environment as described above, then execute the main script:

```bash
python airbnb_train.py
```

## 4. Containerization with Docker

### Building the Docker Container

Build the Docker container using the provided Dockerfile. This encapsulates the application and its environment:

```bash
docker build -t airbnb_price_predictor .
```

### Running the Container

Run the Docker container. This starts the application within an isolated environment:

```bash
docker run -p 5000:5000 airbnb_price_predictor
```

The application is now accessible at `localhost:5000`.

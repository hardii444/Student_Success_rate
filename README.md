# Student_Success_rate
# Ensemble Learning with Voting Regressor

This project demonstrates the use of ensemble learning techniques by combining two regression models: RandomForestRegressor and GradientBoostingRegressor. The ensemble model is built using `VotingRegressor` from the `scikit-learn` library, which takes the predictions from multiple models and averages them to improve prediction accuracy.

## Project Overview

In this project, we train an ensemble model on a given dataset to predict the target values. The two regressors used in the ensemble model are:

- **RandomForestRegressor**: A powerful ensemble learning method that creates multiple decision trees and merges them to get a more accurate and stable prediction.
- **GradientBoostingRegressor**: A boosting algorithm that builds models sequentially, with each new model attempting to correct the errors made by the previous models.

The ensemble model uses `VotingRegressor` to combine these two models and improve performance. After training the model, its performance is evaluated using the Mean Squared Error (MSE) and R-squared (R²) metrics.

## Requirements

To run this project, you'll need the following Python packages:

- **scikit-learn**: Provides the machine learning models and evaluation metrics.
- **numpy** (optional): For numerical operations (if needed in your dataset preprocessing).
- **pandas** (optional): For handling datasets in a DataFrame format.

### Installation

Install the necessary dependencies using pip:

pip install scikit-learn
pip install numpy
pip install pandas



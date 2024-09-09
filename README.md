# Titanic Survival Prediction Project

This project is a machine learning solution for the Kaggle Titanic competition, which aims to predict passenger survival on the Titanic.

## Project Structure

- `main.py`: Script for loading the trained model and making predictions on the test set
- `pipeline.py`: Contains the data preprocessing and model training pipeline
- `data/`: Directory containing the input data files (train.csv and test.csv)
- `model/`: Directory for storing the trained model and prediction output

## Features

- Data preprocessing pipeline
- Feature engineering to create new informative features
- Multiple model comparison (Random Forest, Gradient Boosting, XGBoost)
- Hyperparameter tuning using GridSearchCV
- Model persistence using dill for easy deployment

## Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values in 'Age' and 'Embarked'
- Creating new features from 'Cabin'
- Removing age outliers
- Creating family size and 'is_alone' features
- Extracting titles from names
- Creating name length and ticket-related features
- Categorizing age into groups

## Model Training

The project trains and compares three models:

1. Random Forest Classifier
2. Gradient Boosting Classifier
3. XGBoost Classifier

Hyperparameters for each model are tuned using GridSearchCV with 5-fold cross-validation.

## Usage

1. Ensure you have the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Place the Kaggle competition data files (train.csv and test.csv) in the `data/` directory.

3. Run the pipeline to train the model:
   ```
   python pipeline.py
   ```

4. Make predictions on the test set:
   ```
   python main.py
   ```

5. The predictions will be saved in `model/titanic_predictions.csv`.

## Model Performance

The best performing model and its accuracy score are printed after training. The model, along with metadata, is saved in `titanic_classifier_model.pkl`.


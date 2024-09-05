import dill
import datetime
import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

TRAIN_DATA_PATH = '../data/train.csv'

params_grid = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 8, 12],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 8, 12],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__min_child_weight': [1, 5, 10]
    }
}

def fill_na_age(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['Age'] = df_upd.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
    return df_upd

def fill_na_embarked(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    embarked_mode = df_upd['Embarked'].mode()[0]
    df['Embarked'] = df_upd['Embarked'].fillna(embarked_mode)
    return df_upd

def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['cabin_letter_code'] = df_upd['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'None')
    df_upd['no_of_cabins'] = df_upd['Cabin'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    df_upd = df_upd.drop(columns=['Cabin'], axis=1)
    return df_upd

def remove_age_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    q1 = df_upd['Age'].quantile(0.25)
    q3 = df_upd['Age'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    boundaries = [lower_bound, upper_bound]

    df_upd['Age'] = np.where(df_upd['Age'] > boundaries[1], boundaries[1], df_upd['Age'])
    return df_upd

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['family_size'] = df_upd['SibSp'] + df_upd['Parch'] + 1
    return df_upd

def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['is_alone'] = (df_upd['family_size'] == 1).astype(int)
    return df_upd

def create_title(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['title'] = df_upd['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

    df_upd['title'] = df_upd['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                               'Jonkheer', 'Dona'], 'Rare')
    df_upd['title'] = df_upd['title'].replace('Mlle', 'Miss')
    df_upd['title'] = df_upd['title'].replace('Ms', 'Miss')
    df_upd['title'] = df_upd['title'].replace('Mme', 'Mrs')
    return df_upd

def create_name_length(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['name_length'] = df_upd['Name'].apply(len)
    return df_upd

def create_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['ticket_prefix'] = df_upd['Ticket'].str.extract('([A-Za-z]+)', expand=False).fillna('None')
    return df_upd

def create_ticket_length(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['ticket_length'] = df_upd['Ticket'].apply(len)
    return df_upd

def create_age_category(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['age_category'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 60, 100],
                                labels=['child', 'teen', 'young adult', 'adult', 'middle aged', 'senior'])
    return df_upd

def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df_upd = df_upd.drop(cols_to_drop, axis=1)
    return df_upd

# Defining main pipeline logic
def main():
    # Loading dataset
    df = pd.read_csv(TRAIN_DATA_PATH)
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']

    # Defining categorical and numerical features
    categorical_features = make_column_selector(dtype_include=['object', 'category'])
    numerical_features = ['Age', 'Fare', 'name_length', 'ticket_length']

    # Creating preprocessor pipeline for filling NaN values and creating new features
    preprocessor = Pipeline(steps=[
        ('fill_na_age', FunctionTransformer(fill_na_age)),
        ('fill_na_embarked', FunctionTransformer(fill_na_embarked)),
        ('add_cabin_features', FunctionTransformer(add_cabin_features)),
        ('remove_age_outliers', FunctionTransformer(remove_age_outliers)),
        ('create_family_size', FunctionTransformer(create_family_size)),
        ('create_is_alone', FunctionTransformer(create_is_alone)),
        ('create_title', FunctionTransformer(create_title)),
        ('create_name_length', FunctionTransformer(create_name_length)),
        ('create_ticket_prefix', FunctionTransformer(create_ticket_prefix)),
        ('create_ticket_length', FunctionTransformer(create_ticket_length)),
        ('create_age_category', FunctionTransformer(create_age_category)),
        ('drop_cols', FunctionTransformer(drop_cols))
    ])

    # Defining transformation logic for features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ohe_transformer', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler())
    ])

    # Creating column transformer
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_transformer', categorical_transformer, categorical_features),
        ('numerical_transformer', numerical_transformer, numerical_features)
    ])

    models = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

    # Performing GridSearchCV on multiple models
    best_score = 0
    best_model = None
    for model_name, model in models.items():
        print(f'Tuning hyperparameters for {model_name}...')
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('column_transformer', column_transformer),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(pipe, params_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    print(f'Best Model: {type(best_model.named_steps['classifier']).__name__} with Accuracy: {best_score:.4f}')

    with open('titanic_classifier_model.pkl', 'wb') as f:
        dill.dump({
            'model': best_model,
            'metadata': {
                'name': 'Titanic Survival Classifier',
                'author': 'Vasilii Tokarev',
                'version': '1.1',
                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'type': type(best_model.named_steps['classifier']).__name__,
                'score': best_score
            }
        }, file=f, recurse=True)

if __name__ == '__main__':
    main()

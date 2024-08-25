from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Utils:

    @staticmethod
    def missing_values_percentage(df: pd.DataFrame) -> None:
        """Functions finds complete rows in percentage and rows with missing values along with the percentage and then
        prints out results."""
        missing_values_df_level = (df.notna().all(axis=1).sum() / df.shape[0])
        print(f'Complete rows in percentage: {missing_values_df_level * 100:.2f}%')

        missing_values = ((df.isna().sum() / df.shape[0]) * 100).sort_values(ascending=False).round(2)
        print(f'Missing values in percentage:\n{missing_values[missing_values > 0]}')


    @staticmethod
    def find_outliers_iqr(df:pd.DataFrame, column: pd.DataFrame.columns) -> Any:
        """Functions finds outliers based on IQR. It returns outliers as dataframe and IQR boundaries as a list."""
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        boundaries = [lower_bound, upper_bound]

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        return outliers, boundaries


    @staticmethod
    def create_bar_plot(df: pd.DataFrame, column: pd.DataFrame.columns) -> None:
        """Functions creates a count plot of the selected column in data frame"""
        value_counts = df[column].value_counts().sort_index()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values, order=value_counts.index)

        plt.title(f'Distribution of {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def create_bivar_bar_plot(df: pd.DataFrame, column: pd.DataFrame.columns, target_feature: pd.DataFrame.columns) \
            -> None:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[column], hue=target_feature, data=df)

        plt.title(f'Bar plot of {column} by {target_feature}')
        plt.show()

    @staticmethod
    def create_bivar_violin_plot(df: pd.DataFrame, column: pd.DataFrame.columns, target_feature: pd.DataFrame.columns) \
            -> None:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=target_feature, y=column, data=df)

        plt.title(f'Violin plot of {column} by {target_feature}')
        plt.show()

    @staticmethod
    def categorical_feature_ohe(df: pd.DataFrame, column: pd.DataFrame.columns) -> pd.DataFrame:
        ohe = OneHotEncoder(sparse_output=False)

        ohe.fit(df[[column]])
        ohe_columns = ohe.transform(df[[column]])
        ohe_df = pd.DataFrame(ohe_columns, columns=ohe.get_feature_names_out([column]))

        df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=[column], axis=1)

        return df

    @staticmethod
    def numerical_feature_std(df: pd.DataFrame, column: pd.DataFrame.columns) -> pd.DataFrame:
        std = StandardScaler()

        std.fit(df[[column]])
        std_column = std.transform(df[[column]])

        df[f'{column}_std'] = std_column
        df = df.drop(columns=[column], axis=1)

        return df

    @staticmethod
    def select_best_model(models: dict, x: pd.DataFrame, y: pd.DataFrame) -> Any:
        best_model = None
        best_score = 0
        results = []

        for name, model in models.items():
            cv_scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')
            mean_score = np.mean(cv_scores)
            results.append((name, mean_score))

            print(f'{name}: Mean CV Accuracy equals to {mean_score:.4f}')

            if mean_score > best_score:
                best_score = mean_score
                best_model = model

        print(f'Best model: {best_model.__class__.__name__}')

        return best_model, results

    @staticmethod
    def hyper_params_tuning(model: dict, x: pd.DataFrame, y: pd.DataFrame, param_grid: dict) -> Any:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x, y)

        print(f'Best parameters found: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_:.4f}')

        return grid_search.best_params_

from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.core.interchange import column


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

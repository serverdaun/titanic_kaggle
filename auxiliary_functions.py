from typing import Tuple, Any, List
import pandas as pd
from pandas import Series, DataFrame


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
    def find_outliers_iqr(df:pd.DataFrame, column: pd.DataFrame.columns):
        """Functions finds outliers based on IQR. It returns outliers as dataframe and IQR boundaries as a list."""
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        boundaries = [lower_bound, upper_bound]

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        return outliers, boundaries

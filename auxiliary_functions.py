import pandas as pd


class Utils:

    @staticmethod
    def missing_values_percentage(df: pd.DataFrame) -> None:
        """Functions finds complete rows in percentage and rows with missing values along with the percentage and then
        prints out results"""
        missing_values_df_level = (df.notna().all(axis=1).sum() / df.shape[0])
        print(f'Complete rows in percentage: {missing_values_df_level * 100:.2f}%')

        missing_values = ((df.isna().sum() / df.shape[0]) * 100).sort_values(ascending=False).round(2)
        print(f'Missing values in percentage:\n{missing_values[missing_values > 0]}')


    @staticmethod
    def find_outliers_iqr(df:pd.DataFrame, column: pd.DataFrame.columns) -> pd.DataFrame:
        """Functions finds outliers based on IQR and returns them as dataframe"""
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        outliers = df[(df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))]
        return outliers

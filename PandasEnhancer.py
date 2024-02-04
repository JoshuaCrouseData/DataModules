import pandas as pd
import pandas_flavor as pf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split


class DataFrameManipulator:
    def __init__(self, data, columns):
        self.df = pd.DataFrame(data, columns=columns)

    @pf.register_dataframe_method
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df

    @pf.register_dataframe_method
    def fill_na(self, value):
        self.df = self.df.fillna(value)
        return self.df

    @pf.register_dataframe_method
    def get_summary(self):
        return self.df.describe()
    
    @pf.register_dataframe_method
    def normalize(self, column):
        self.df[column] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())
        return self.df
    
    @pf.register_dataframe_method
    def interpolate(self):
        self.df = self.df.interpolate()
        return self.df
    
    @pf.register_dataframe_method
    def encode(self, column):
        le = LabelEncoder()
        self.df[column] = le.fit_transform(self.df[column])
        return self.df

    @pf.register_dataframe_method
    def add_interaction(self, column1, column2):
        self.df[f'{column1}_{column2}'] = self.df[column1] * self.df[column2]
        return self.df
    
    @pf.register_dataframe_method
    def t_test(self, column, group_column):
        group1 = self.df[self.df[group_column] == 0][column]
        group2 = self.df[self.df[group_column] == 1][column]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return t_stat, p_value

    # Create charts directly from a dataframe
    @pf.register_dataframe_method
    def plot_histogram(self, column, title="Histogram", bins=10):
        self.df[column].hist(bins=bins)
        plt.title(title)
        plt.show()

    @pf.register_dataframe_method
    def plot_scatter(self, x_column, y_column, title="Scatter Plot"):
        self.df.plot.scatter(x=x_column, y=y_column)
        plt.title(title)
        plt.show()

    @pf.register_dataframe_method
    def plot_line(self, x_column, y_column, title="Line Plot"):
        self.df.plot.line(x=x_column, y=y_column)
        plt.title(title)
        plt.show()

    @pf.register_dataframe_method
    def plot_bar(self, x_column, y_column, title="Bar Chart"):
        self.df.plot.bar(x=x_column, y=y_column)
        plt.title(title)
        plt.show()

    @pf.register_dataframe_method
    def plot_pie(self, column, title="Pie Chart"):
        self.df[column].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(title)
        plt.show()

    def add_data(self, data_dict):
        new_row = pd.Series(data_dict)
        self.df = self.df.append(new_row, ignore_index=True)

    def sort_by_column(self, column, ascending=True):
        self.df = self.df.sort_values(by=column, ascending=ascending)
        return self.df

    def get_dataframe(self):
        return self.df
    
    @pf.register_dataframe_method
    def remove_outliers(self, column):
        z_scores = stats.zscore(self.df[column])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3)
        self.df = self.df[filtered_entries]
        return self.df

    @pf.register_dataframe_method
    def convert_dtype(self, column, dtype):
        self.df[column] = self.df[column].astype(dtype)
        return self.df

    @pf.register_dataframe_method
    def plot_heatmap(self):
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()

    @pf.register_dataframe_method
    def split_data(self, test_size=0.2, random_state=None):
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train, test

    @pf.register_dataframe_method
    def standardize_column(self, column):
        scaler = StandardScaler()
        self.df[column] = scaler.fit_transform(self.df[[column]])
        return self.df

    @pf.register_dataframe_method
    def create_polynomial_features(self, column, degree=2):
        """
        Creates polynomial features of a specified degree for a given column.
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree)
        transformed = poly.fit_transform(self.df[[column]])
        # Add polynomial feature columns to dataframe
        for i in range(1, degree + 1):
            self.df[f'{column}^ {i}'] = transformed[:, i]
        return self.df
    
    @pf.register_dataframe_method
    def bin_column(self, column, bins, labels=None, right=True):
        """
        Bins continuous data into discrete intervals.
        """
        self.df[column] = pd.cut(self.df[column], bins=bins, labels=labels, right=right)
        return self.df

    @pf.register_dataframe_method
    def one_hot_encode(self, column):
        """
        Applies one-hot encoding to a categorical column.
        """
        one_hot = pd.get_dummies(self.df[column], prefix=column)
        self.df = self.df.drop(column, axis=1)
        self.df = pd.concat([self.df, one_hot], axis=1)
        return self.df
    
    @pf.register_dataframe_method
    def plot_correlation_heatmap(self, cmap='viridis'):
        """
        Plots a heatmap of the correlation matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap=cmap)
        plt.title("Correlation Heatmap")
        plt.show()

    @pf.register_dataframe_method
    def plot_boxplot(self, column):
        """
        Creates a box plot for the specified column.
        """
        sns.boxplot(x=self.df[column])
        plt.title(f"Box Plot of {column}")
        plt.show()

    @pf.register_dataframe_method
    def plot_violinplot(self, column):
        """
        Creates a violin plot for the specified column.
        """
        sns.violinplot(x=self.df[column])
        plt.title(f"Violin Plot of {column}")
        plt.show()

    @pf.register_dataframe_method
    def save_to_file(self, filename, format='csv', index=False):
        """
        Saves the DataFrame to a file. Supports CSV, Excel, and JSON formats.
        """
        if format == 'csv':
            self.df.to_csv(filename, index=index)
        elif format == 'excel':
            self.df.to_excel(filename, index=index)
        elif format == 'json':
            self.df.to_json(filename, orient='records', lines=True)
        else:
            raise ValueError("Unsupported file format. Choose 'csv', 'excel', or 'json'.")
        return self.df
    
    @pf.register_dataframe_method
    def merge_with(self, other_df, on, how='inner'):
        """
        Merges the current DataFrame with another DataFrame.
        """
        self.df = pd.merge(self.df, other_df, on=on, how=how)
        return self.df
    
    @pf.register_dataframe_method
    def drop_columns(self, columns):
        """
        Drops specified columns from the DataFrame.
        """
        self.df = self.df.drop(columns, axis=1)
        return self.df
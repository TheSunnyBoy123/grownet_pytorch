import pandas as pd
import csv

class Table:
    def __init__(self, path):
        self.value = pd.read_csv(path, index_col=0)
        self.path = path
        self.X = self.value.iloc[:, :-1]
        self.y = self.value.iloc[:, -1]
        self.columns = self.value.columns
        self.train_columns = self.columns[:-1]
        self.target_column = self.columns[-1]
        self.shape = self.value.shape

    def __str__(self):
        return str(self.value)
    
    def non_integer_columns(self):
        return self.value.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    def integer_columns(self):
        return self.value.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def encode(self, columns):
        for column in columns:
            self.value[column] = self.value[column].astype('category').cat.codes
        self.X = self.value.iloc[:, :-1]
        self.y = self.value.iloc[:, -1]
        self.columns = self.value.columns
        self.train_columns = self.columns[:-1]
        self.target_column = self.columns[-1]
        self.shape = self.value.shape
        return self.value

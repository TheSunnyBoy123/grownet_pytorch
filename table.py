import pandas as pd
import csv

class Table:
    def __init__(self, path):
        self.value = pd.read_csv(path)
        self.path = path
        self.X = self.value.iloc[:, :-1]
        self.y = self.value.iloc[:, -1]
        self.columns = self.value.columns
        self.train_columns = self.columns[:-1]
        self.target_column = self.columns[-1]
        self.shape = self.value.shape

    def __str__(self):
        return str(self.value)
    

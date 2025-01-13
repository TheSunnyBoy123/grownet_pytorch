import pandas as pd
from table import Table

table = Table("train.csv")
print(table.non_integer_columns())
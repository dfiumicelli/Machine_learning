import pandas as pd
import os

pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv(os.path.join('datasets', 'auto-mpg.csv'))
print(df.head())
print(df.describe())
print(df.info())
print(df[df['horsepower'] == '?'])

#df = df.drop(df[df['horsepower'] == '?'].index)


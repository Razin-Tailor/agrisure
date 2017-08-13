import pandas as pd

df = pd.read_csv('ObservationData.csv')

print(df['crop'].unique())
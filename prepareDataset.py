import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import xlrd

df = pd.read_csv('ObservationData.csv')
dfRain = pd.read_excel('list_of_cities.xls')
dfRainN = dfRain.rename(columns={'Name of City':'location'})
# print(dfRain['State'].unique())

merged_df = pd.merge(df,dfRainN,on='location', how='inner')
# print(merged_df.head())
# print(merged_df.columns.values)

dfRainData = pd.read_csv('metereological_subdivisionwise_2010_2011.csv')

dfRainData['mean'] = dfRainData.mean(axis=1)

# print(dfRainData.columns.values)
sliced_df_rain = dfRainData.ix[:,['Sub-division ','mean']]
# print(sliced_df_rain.head(10))
# print(listN, len(list), len(listN))
sliced_df_rain = sliced_df_rain.rename(columns={'Sub-division ':'State'})

final_merged_df = pd.merge(merged_df,sliced_df_rain,on='State', how='inner')
# print(final_merged_df.head())

final_merged_df.drop('S.No', axis=1, inplace=True)
final_merged_df.drop('Population (2011)', axis=1, inplace=True)
final_merged_df.drop('Type', axis=1, inplace=True)
final_merged_df.drop('Population class', axis=1, inplace=True)


print(final_merged_df.head())
print(final_merged_df.columns.values)



done = final_merged_df.to_csv('FinalDataset.csv')

print("############ DONE ##########")
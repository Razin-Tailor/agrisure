import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

rain_data  = pd.read_csv('metereological_subdivisionwise_2010_2011.csv')
df = pd.read_csv('ObservationData.csv')
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
wide_df = pd.DataFrame()
# print(rain_data.head())
# print(crop_data.head())

# print(rain_data.columns)
# print(crop_data.columns)

# rain_data2 = rain_data[rain_data['Sub-division '].str.contains('^Gujarat')]

# print(rain_data2.shape)
# list_cities = ['Ahmadabad','Surat','Vadodara','Rajkot','Anand','Bhavnagar','Jamnagar','c','Junagadh','Navsari']



############# Gujarat Crop Major Cities Data ##################
# crop_data2 = crop_data[crop_data['location'].isin(list_cities)]
# print(crop_data2.head())

# crop_data2010 = crop_data2[crop_data2['Date'] == 2010]
# print(crop_data2010.head())

def dynamic_model(season, param):
	global df
	df = df[df['season'] == season]
	df.drop('season', axis=1, inplace=True)
	df = df[df['indicator'] == param]
	df.drop('indicator', axis=1, inplace=True)
	if df['crop'].isnull().sum() > 0:
		df = df.dropna(axis = 0, subset = ['crop'])

	# print("After processing-------",len(df['crop'].unique()))
	crops = (df.groupby(by = ['crop'])['Value'].sum().reset_index().rename(columns = {'Value': 'total_crop_value'})[['crop', 'total_crop_value']])

	# print(crops.head())

	df_with_crops = df.merge(crops, left_on = 'crop', right_on = 'crop', how = 'left')
	# print(df_with_crops.head())
	print(df_with_crops.describe())
	# print("Total crop value", crops['total_crop_value'].describe())
	quantile = crops['total_crop_value'].quantile(np.arange(.9, 1, .01)).values.tolist()
	print(quantile)
	print crops['total_crop_value'].quantile(np.arange(.9, 1, .01))

	# popularity_threshold = 310977
	popularity_threshold = quantile[3]

	df_popular_crops = df_with_crops.query('total_crop_value >= @popularity_threshold')
	print("df_of_popular_crop", df_popular_crops.head())
	in_data = df_popular_crops
	print("In-data---", in_data.head(), len(in_data['crop'].unique()))
	if not in_data[in_data.duplicated(['location', 'crop'])].empty:
		initial_rows = in_data.shape[0]

		print 'Initial dataframe shape {0}'.format(in_data.shape)
		in_data = in_data.drop_duplicates(['location', 'crop'])
		current_rows = in_data.shape[0]
		print 'New dataframe shape {0}'.format(in_data.shape)
		print 'Removed {0} rows'.format(initial_rows - current_rows)
	else:
		print("consistent data")
	global wide_df
	wide_df = in_data.pivot(index = 'crop', columns = 'location', values = 'Value').fillna(0)
	print(wide_df.head(), wide_df.shape)
	print((wide_df.columns.tolist()))
	wide_df_sparse = csr_matrix(wide_df.values)

	#### Fitting Model ####

	
	model_knn.fit(wide_df_sparse)


# dynamic_model('Rabi','Yield (Tonnes/Hect.)')

# query_index = np.random.choice(wide_df.shape[0])
# distances, indices = model_knn.kneighbors(wide_df.iloc[query_index, :].reshape(1, -1), n_neighbors = 4)

# for i in range(0, len(distances.flatten())):
# 	if i == 0:
# 		print 'Recommendations for {0}:\n'.format(wide_df.index[query_index])
# 	else:
# 		print '{0}: {1}, with distance of {2}:'.format(i, wide_df.index[indices.flatten()[i]], distances.flatten()[i])



	
	# Morvi,
# Gandhidham
# Bharuch
# Porbandar	
# Mahesana
# Bhuj
# Veraval
# Surendranagar
# Valsad
# Vapi
# Godhra
# Palanpur
# Anklesvar
# Patan
# Dahod
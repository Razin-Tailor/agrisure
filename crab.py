# from scikits.crab.models import MatrixPreferenceDataModel
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
# from scikits.crab.metrics import pearson_correlation
# from scikits.crab.similarities import UserSimilarity
# from crab.recommenders.knn import UserBasedRecommender
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv('ObservationData.csv')

dfN = pd.read_csv('2007_ProductionInTonnesBySeasonAndCropsCleaned.csv')
dfTest = pd.read_csv('2007_ProductionInTonnesBySeasonAndCrops.csv', names=['id','location','crop','season','value'])
# print(len(dfTest['location'].unique()))
locations = dfTest['location'].unique()
print(df.index)
df = df[df.season == 'Total']
df.drop('season', axis=1, inplace=True)
print(df['Date'].dtype)
df = df[df['Date'] == 2007]
df.drop('Date', axis=1, inplace=True)
df = df[df['indicator'] == 'Production (In Tonnes)']
df.drop('indicator', axis=1, inplace=True)
df.drop('Unit', axis=1, inplace=True)

print(df.head(), len(df['crop'].unique()))

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
wide_df = pd.DataFrame()


# col = ['Date', 'season']
# dfnew = pd.read_csv('ObservationData.csv')
# dfnew = dfnew[dfnew.indicator == 'Production (In Tonnes)']
# print(dfnew.head(20))
# dfnew = df.convert_objects(convert_numeric=True)
# dfnew.fillna(0, inplace=True)
# print(dfnew.head(20))
# dfnew = dfnew.groupby(['Date', 'season'])
# print(dfnew.head())

# dfnew[['Value', 'season']].plot(kind='line')
# plt.show()

# rainfall = pd.read_csv('metereological_subdivisionwise_2010_2011.csv')
# # print(rainfall.columns.values)
# print(rainfall['Sub-division '].unique(), len(rainfall['Sub-division '].unique()))

# def handle_non_numeric_data(df):
# 	columns = df.columns.values
# 	for column in columns:
# 		text_digit_vals = {}
# 		def convert_to_int(val):
# 			return text_digit_vals[val]

# 		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
# 			column_contents = df[column].values.tolist()
# 			unique_elements = set(column_contents)
# 			x = 0
# 			for unique in unique_elements:
# 				if unique not in text_digit_vals:
# 					text_digit_vals[unique] = x
# 					x = x + 1
# 			df[column] = list(map(convert_to_int, df[column]))
# 	return df

# clean_df = handle_non_numeric_data(dfnew)
# print(clean_df.head(20))

# split_index = int(rows * 0.9)
# # Use indices to separate the data
# df_train = df[0:split_index]
# df_test = df[split_index:].reset_index(drop=True)

# print(df.head(20))
# print(df.head(60).ix[0:,0:2])

if df['crop'].isnull().sum() > 0:
	df = df.dropna(axis = 0, subset = ['crop'])

print("After processing-------",len(df['crop'].unique()))
crops = (df.groupby(by = ['crop'])['Value'].sum().reset_index().rename(columns = {'Value': 'total_crop_value'})[['crop', 'total_crop_value']])

# print(crops.head())

df_with_crops = df.merge(crops, left_on = 'crop', right_on = 'crop', how = 'left')
# print(df_with_crops.head())

print("Total crop value", crops['total_crop_value'].describe())
print crops['total_crop_value'].quantile(np.arange(.9, 1, .01))

popularity_threshold = 310977
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

wide_df = in_data.pivot(index = 'crop', columns = 'location', values = 'Value').fillna(0)
print(wide_df.head(), wide_df.shape)
# print((wide_df.columns.tolist()))
wide_df_sparse = csr_matrix(wide_df.values)

#### Fitting Model ####

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_df_sparse)


def print_artist_recommendations(query_crop, df_matrix, knn_model, k):
    """
    Inputs:
    query_artist: query crop name
    artist_plays_matrix: artist play count dataframe (not the sparse one, the pandas dataframe)
    knn_model: our previously fitted sklearn knn model
    k: the number of nearest neighbors.
    
    Prints: Artist recommendations for the query artist
    Returns: None
    """

    query_index = None
    ratio_tuples = []
    # print(df_matrix.index)
    for i in df_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_crop.lower())
        if ratio >= 75:
            current_query_index = df_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples])
    print ratio_tuples

    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2]
        print query_index # get the index of the best artist match in the data
    except:
        print 'Your artist didn\'t match any artists in the data. Try again'
        return None
    print(df_matrix.iloc[query_index, :])
    distances, indices = knn_model.kneighbors(df_matrix.iloc[query_index, :].reshape(1, -1), n_neighbors = k + 1)
    print(distances.flatten(), indices)
    print(df_matrix.index[query_index])
    # print(df_matrix.index[indices.flatten()[i]])
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print 'Recommendations for {0}:\n'.format(df_matrix.index[query_index])
        else:
            print(df_matrix.index[indices.flatten()[i]])
            print(distances.flatten()[i])
            print '{0}: {1}, with distance of {2}:'.format(i, df_matrix.index[indices.flatten()[i]], distances.flatten()[i])
    return None

############### Recommendations ###############

# query_index = np.random.choice(wide_df.shape[0])
# distances, indices = model_knn.kneighbors(wide_df.iloc[query_index, :].reshape(1, -1), n_neighbors = 10)

# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         print 'Recommendations for {0}:\n'.format(wide_df.index[query_index])
#     else:
#         print '{0}: {1}, with distance of {2}:'.format(i, wide_df.index[indices.flatten()[i]], distances.flatten()[i])

# for i in range(0, len(locations)):
# 	print(i, locations[i])

print_artist_recommendations('Jowar',wide_df,model_knn,5)


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



from flask import Flask, render_template, request, jsonify

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from crab import *
# import matplotlib.pyplot as plt


app = Flask(__name__)

db_name = 'mydb'
client = None
db = None

pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv('ObservationData.csv')
dfnew = df[df['indicator']=='Production (In Tonnes)']
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
wide_df = pd.DataFrame()
port = int(os.getenv('PORT', 8080))

def prepare_model():
    global df
    if df['crop'].isnull().sum() > 0:
        df = df.dropna(axis = 0, subset = ['crop'])

    # print("After processing-------",len(df['crop'].unique()))
    crops = (df.groupby(by = ['crop'])['Value'].sum().reset_index().rename(columns = {'Value': 'total_crop_value'})[['crop', 'total_crop_value']])

    # print(crops.head())

    df_with_crops = df.merge(crops, left_on = 'crop', right_on = 'crop', how = 'left')
    # print(df_with_crops.head())

    # print("Total crop value", crops['total_crop_value'].describe())
    print (crops['total_crop_value'].quantile(np.arange(.9, 1, .01)))

    popularity_threshold = 310977
    df_popular_crops = df_with_crops.query('total_crop_value >= @popularity_threshold')
    # print("df_of_popular_crop", df_popular_crops.head())
    in_data = df_popular_crops
    # print("In-data---", in_data.head(), len(in_data['crop'].unique()))
    if not in_data[in_data.duplicated(['location', 'crop'])].empty:
        initial_rows = in_data.shape[0]

        print( 'Initial dataframe shape {0}'.format(in_data.shape))
        in_data = in_data.drop_duplicates(['location', 'crop'])
        current_rows = in_data.shape[0]
        print( 'New dataframe shape {0}'.format(in_data.shape))
        print( 'Removed {0} rows'.format(initial_rows - current_rows))
    else:
        print("consistent data")
    global wide_df
    wide_df = in_data.pivot(index = 'crop', columns = 'location', values = 'Value').fillna(0)
    # print(wide_df.head(), wide_df.shape)
    # print((wide_df.columns.tolist()))
    wide_df_sparse = csr_matrix(wide_df.values)

    #### Fitting Model ####

    
    model_knn.fit(wide_df_sparse)

def get_crop_recommendations(query_crop, df_matrix, knn_model, k):
    print(df_matrix.head())
    query_index = None
    ratio_tuples = []
    # print(df_matrix.index)
    for i in df_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_crop.lower())
        if ratio >= 75:
            current_query_index = df_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print ('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    print (ratio_tuples)

    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2]
        print (query_index) # get the index of the best artist match in the data
    except:
        print ('Your artist didn\'t match any artists in the data. Try again')
        return None
    print(df_matrix.iloc[query_index, :])
    distances, indices = knn_model.kneighbors(df_matrix.iloc[query_index, :].reshape(1, -1), n_neighbors = k + 1)
    print(distances.flatten(), indices)
    print(df_matrix.index[query_index])
    # print(df_matrix.index[indices.flatten()[i]])
    dict = {}
    for i in range(0, len(distances.flatten())):
        list = []
        if i == 0:
            list.append(df_matrix.index[query_index])
            print ('Recommendations for {0}:\n'.format(df_matrix.index[query_index]))
            return list
        else:
            # print(df_matrix.index[indices.flatten()[i]])
            # print(distances.flatten()[i])
            list.append((df_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
            print ('{0}: {1}, with distance of {2}:'.format(i, df_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
        return list
    return None





@app.route("/")
def graph(chartID = 'chart_ID', chart_type = 'line', chart_height = 500):
    global df
    global dfnew
    chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
    series = [{"name": 'Production', "data": dfnew['Value'] }]
    title = {"text": 'Dates'}
    xAxis = {"categories": (df['Date'])}
    yAxis = {"title": {"text": 'Production'}}
    return render_template('index.html', chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
 
@app.route('/recomend', methods=['POST'])
def getData():
    data = request.get_json()
    crop = data['crop']
    recommendations = get_crop_recommendations(crop,wide_df, model_knn, 5)
    return jsonify({'status': recommendations})


@app.route('/dynamic', methods=['POST'])
def rec():
    data = request.get_json()
    season = data['season']
    param = data['param']
    dynamic_model(season,param)

    query_index = np.random.choice(wide_df.shape[0])
    distances, indices = model_knn.kneighbors(wide_df.iloc[query_index, :].reshape(1, -1), n_neighbors = 4)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print( 'Recommendations for {0}:\n'.format(wide_df.index[query_index]))
        else:
            print( '{0}: {1}, with distance of {2}:'.format(i, wide_df.index[indices.flatten()[i]], distances.flatten()[i]))
    return jsonify({'status':'success'})

if __name__ == '__main__':
    prepare_model()
    app.run(host='0.0.0.0', port=port, debug=True)

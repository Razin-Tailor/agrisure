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
wide_df_rain = pd.DataFrame()
port = int(os.getenv('PORT', 8080))


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
    list = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            # list.append(df_matrix.index[query_index])
            print ('Recommendations for {0}:\n'.format(df_matrix.index[query_index]))
            # return list
        else:
            # print(df_matrix.index[indices.flatten()[i]])
            # print(distances.flatten()[i])
            list.append((df_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
            print ('{0}: {1}, with distance of {2}:'.format(i, df_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
        # return list
    list.sort(key = lambda x: x[1])
    return list

@app.route("/")
def graph(chartID = 'chart_ID', chart_type = 'line', chart_height = 500):
    # global df
    # global dfnew
    # chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
    # series = [{"name": 'Production', "data": dfnew['Value'] }]
    # title = {"text": 'Dates'}
    # xAxis = {"categories": (df['Date'])}
    # yAxis = {"title": {"text": 'Production'}}
    # return render_template('index.html', chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
    return render_template('dashboard-2.html')
 
@app.route('/recomend', methods=['POST'])
def getData():
    data = request.get_json()
    crop = data['crop']
    recommendations = get_crop_recommendations(crop,wide_df, model_knn, 5)
    return jsonify({'status': recommendations})

@app.route('/dynamic', methods=['POST'])
def rec():
    data = request.get_json()
    crop = data['crop']
    season = data['season']
    param = data['param']
    dynamic_model(season,param)
    recommendations = get_crop_recommendations(crop,wide_df, model_knn, 5)
    return jsonify({'suggestion' : recommendations})

@app.route('/dynamicrain', methods=['POST'])
def rec_rain():
    data = request.get_json()
    crop = data['crop']
    season = data['season']
    param = data['param']
    if param == 'mean':
        dynamic_model(season,param)
        recommendations = get_crop_recommendations(crop,wide_df, model_knn, 5)
        return jsonify({'suggestion' : recommendations})
    else:
        return jsonify({'status': 'failure'})

if __name__ == '__main__':
    # prepare_model()
    app.run(host='0.0.0.0', port=port, debug=True)

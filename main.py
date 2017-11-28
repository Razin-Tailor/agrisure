from config import *

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

def preprocess(df):
    df = pd.read_csv('ObservationData.csv')
    df = df[df['season'] == season]
    df.drop('season', axis=1, inplace=True)
    df = df[df['indicator'] == param]
    df.drop('indicator', axis=1, inplace=True)
    if df['crop'].isnull().sum() > 0:
        df = df.dropna(axis = 0, subset = ['crop'])
        return df
    else:
        return df

def dynamic_model(season, param):
    df = preprocess(df) 
    # print("After processing-------",len(df['crop'].unique()))
    crops = (df.groupby(by = ['crop'])['Value'].sum().reset_index().rename(columns = {'Value': 'total_crop_value'})[['crop', 'total_crop_value']])

    # print(crops.head())

    df_with_crops = df.merge(crops, left_on = 'crop', right_on = 'crop', how = 'left')
    # print(df_with_crops.head())
    print(df_with_crops.describe())
    # print("Total crop value", crops['total_crop_value'].describe())
    quantile = crops['total_crop_value'].quantile(np.arange(.9, 1, .01)).values.tolist()
    print(quantile)
    print( crops['total_crop_value'].quantile(np.arange(.9, 1, .01)))

    # popularity_threshold = 310977
    popularity_threshold = quantile[3]

    df_popular_crops = df_with_crops.query('total_crop_value >= @popularity_threshold')
    print("df_of_popular_crop", df_popular_crops.head())
    in_data = df_popular_crops
    print("In-data---", in_data.head(), len(in_data['crop'].unique()))
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
    print(wide_df.head(), wide_df.shape)
    print((wide_df.columns.tolist()))
    wide_df_sparse = csr_matrix(wide_df.values)

    #### Fitting Model ####
    
    model_knn.fit(wide_df_sparse)


@app.route("/")
def graph(chartID = 'chart_ID', chart_type = 'line', chart_height = 500):
    return "Hello World"
 
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
    print(season,param)
    dynamic_model(season,param)
    recommendations = get_crop_recommendations(crop,wide_df, model_knn, 3)
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
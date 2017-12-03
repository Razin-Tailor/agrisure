# AgriSure

AgriSure is a Crop-Recommender-System. It is built with the vision of updating the traditional crop cultivation approach. With the utilisation of Historical Data, It takes into account various factors and provides the best 3 suggestions of crops for farmers to cultivate in a perticular season.

## Pre-requisites


```sh
$ git clone https://gitlab.com/Razin-Tailor/agrisure.git
$ cd agrisure-backend

```

Compile and Run the Android Application


## Running Server Locally


```sh
$ git clone https://gitlab.com/Razin-Tailor/agrisure-backend.git
$ cd agrisure-backend

$ pip install -r requirements.txt

$ python main.py

```

Your app should now be running on [0.0.0.0:8080](http://0.0.0.0:8080/).

## Approach

### Filtering to Only Popular Crops

Since we’re going to be doing item-based collaborative filtering, our recommendations will be based on location patterns in growing crops. Lesser grown crops will have *params* from fewer locations, making the pattern more noisy. This would probably result in bad recommendations (or at least ones highly sensitive to an individual location who loves one obscure crop. To avoid this problem, we’ll just look at the popular crops.

To find out which crops are popular, we need to know the total *param* count of every crop. Since our location *param* count data has one row per crop per location, we need to aggregate it up to the crop level. With pandas, we can group by the crop name and then calculate the sum of the *param* column for every crop. If the crop-name variable is missing, our future reshaping and analysis won’t work. So I’ll start by removing rows where the crop name is missing just to be safe.

Hence we merge the total *param* count data into the location activity data, giving us exactly what we need to filter out the lesser known crops.

### Picking a threshold for popular crops

With nearly 300,000 different crops, it’s almost a guarantee most crops have less *params*

So we filter out the data with the right threshold

Before doing any analysis, we should make sure the dataset is internally consistent. Every location should only have a *param* count variable once for each crop. So we’ll check for instances where rows have the same location and crop-name values.

### Implemeting the Nearest Neighbor Model

#### Reshaping the Data

For K-Nearest Neighbors, we want the data to be in an m x n array, where m is the number of crops and n is the number of locations. To reshape the dataframe, we’ll pivot the dataframe to the wide format with crops as rows and location as columns. Then we’ll fill the missing observations with 0s since we’re going to be performing linear algebra operations (calculating distances between vectors). Finally, we transform the values of the dataframe into a scipy sparse matrix for more efficient calculations.

#### Fitting the Model

Time to implement the model. We’ll initialize the NearestNeighbors class as model_knn and fit our sparse matrix to the instance. By specifying the metric = cosine, the model will measure similarity between crop vectors by using cosine similarity.
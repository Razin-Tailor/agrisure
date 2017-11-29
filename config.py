from imports import *

app = Flask(__name__)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv('Datasets/ObservationData.csv')
dfnew = df[df['indicator']=='Production (In Tonnes)']
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
wide_df = pd.DataFrame()
wide_df_rain = pd.DataFrame()
port = int(os.getenv('PORT', 8080))

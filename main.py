import json
import sys
import numpy as np
import pandas
import pylab as plt
#from bson import json_util
from flask import Flask,render_template
from scipy.spatial.distance import cdist
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob
from flask import request
input_file = pandas.read_csv('Crime_Data_County.csv', low_memory=False)
import datetime
loadingVector = {}

columns = ['Murders', 'Rapes', 'Robberies', 'Assaults', 'Burglaries', 'Larencies', 'Thefts', 'Arsons', 'Population']

minmaxscaler = MinMaxScaler()
input_file[columns]=minmaxscaler.fit_transform(input_file[columns])

features = input_file[columns]
data = np.array(features)
random_samples = []
for j in range(400):
    random_samples.append(data[j])

eigenValues = []
eigenVectors = []

def plotElbow():
    # print("Plotting Elbow plot");
    global input_file
    features = input_file[columns]

    k = range(1, 11)

    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]

    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]

    kidx = 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, avg_within, 'g*-')
    ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()


def clustering():
    # Clustering the data
    # print("Clustering data with K = 4");
    global input_file
    features = input_file[columns]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    labels = kmeans.labels_
    input_file['kcluster'] = pandas.Series(labels)


def generate_eig_values(data):
    global eigenValues
    global eigenVectors

    centered_matrix = data - np.mean(data, axis=0)
    cov = np.dot(centered_matrix.T, centered_matrix)
    eigenValues, eigenVectors = np.linalg.eig(cov)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    eigenValues[0] = 2.84578214;
    eigenValues = eigenValues * 1.5

def plot_intrinsic_dimensionality_pca(data, k):
    # print("Inside plot_intrinsic_dimensionality_pca")
    global loadingVector
    global eigenValues
    global eigenVectors

    idx = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:, idx]
    squaredLoadings = []
    ftrCount = len(eigenVectors)
    for ftrId in range(0, ftrCount):
        loadings = 0
        for compId in range(0, k):
            loadings = loadings + eigenVectors[compId][ftrId] * eigenVectors[compId][ftrId]
        loadingVector[columns[ftrId]] = loadings
        squaredLoadings.append(loadings)

    # print("Return Squareloadings")
    # print(loadingVector)
    return squaredLoadings

# plotElbow()
clustering()

generate_eig_values(data)
squared_loadings = plot_intrinsic_dimensionality_pca(data, 3)
imp_fetures = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)
print(imp_fetures)


MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'crime_db'
CRIME_DATA_STATE_COLLECTION = 'crime_data_state'
CRIME_DATA_COUNTY_COLLECTION = 'crime_data_county'
CRIME_REPORT_COLLECTION = 'crime_report'
CRIME_ANALYSIS_COLLECTION = 'test'

STATE_DATA_FIELDS = {'State': True, 'Murders': True, 'Rapes': True, 'Robberies': True, 'Assaults': True, 'Burglaries': True, 'Larencies': True, 'Thefts': True, 'Arsons': True,
                     'Murders_Rate': True, 'Rapes_Rate': True, 'Robberies_Rate': True, 'Assaults_Rate': True,
                     'Burglaries_Rate': True, 'Larencies_Rate': True, 'Thefts_Rate': True, 'Arsons_Rate': True, 'Population': True,'id': True, '_id': False}
COUNTY_DATA_FIELDS = {'rate': True, 'County Name': True, 'Murders': True, 'Rapes': True, 'Robberies': True, 'Assaults': True, 'Burglaries': True, 'Larencies': True, 'Thefts': True, 'Arsons': True,
                      'Murders_Rate': True, 'Rapes_Rate': True, 'Robberies_Rate': True, 'Assaults_Rate': True,
                      'Burglaries_Rate': True, 'Larencies_Rate': True, 'Thefts_Rate': True, 'Arsons_Rate': True, 'Population': True, 'FIPS_ST': True, 'FIPS_CTY': True, 'id': True, '_id': False}
REPORT_FIELDS = {'State Abbr': True, 'Year': True, 'Crime Solved': True, 'Victim Sex': True, 'Victim Age': True, 'Victim Race': True, 'Perpetrator Sex': True, 'Perpetrator Age': True, 'Perpetrator Race': True, 'Weapon': True, '_id': False}
ANALYSIS_FIELDS = {'id': True, 'rate': True, '_id': False}

# connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
# state_data_collection = connection[DBS_NAME][CRIME_DATA_STATE_COLLECTION]
# county_data_collection = connection[DBS_NAME][CRIME_DATA_COUNTY_COLLECTION]
# report_collection = connection[DBS_NAME][CRIME_REPORT_COLLECTION]
# analysis_collection = connection[DBS_NAME][CRIME_ANALYSIS_COLLECTION]

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("dashboard.html")
    # return render_template("test_ts.html")
    # return render_template("test_pie.html")


@app.route("/hospitals", methods = ['POST','GET'])
def gethospitals():
    #df = pandas.read_csv('Crime_Data_State.csv')
    df = pandas.read_csv('HospitalBedsIndia.csv')
    cols = json.dumps(list(df.columns))
    df = df.fillna(int(0))
    rows = json.dumps(df.to_dict(orient='records'), indent=2)
    data = { 'rows': rows, 'cols': cols}
    return data

@app.route("/getconfirmedcases",methods = ['POST','GET'])
def stackedarea():
    start,end = 25,40
    import pandas as pd
    df = pd.read_csv('covid_19_india.csv')
    states = df['State/UnionTerritory'].unique()
    regions,confirmed = [df[df['State/UnionTerritory'] == s] for s in states],[]
    for i,s in enumerate(states):
        cnfrmed = list(regions[i][end-1:end].Confirmed)
        if len(cnfrmed) == 0: continue
        confirmed.append((cnfrmed[0],s,i))
    confirmed = sorted(confirmed,reverse=True)
    #Top 10 state ids
    stateids = [k for i,j,k in confirmed[:10]]
    states = [j for i,j,k in confirmed[:10]]
    cnt = [ list(regions[i][start:end].Confirmed) for i in stateids]
    df = pd.DataFrame()
    for i,s in enumerate(states):
        df[s] = cnt[i]
    cols = json.dumps(list(df.columns))
    df['total'] = np.sum(cnt,axis=0)
    df['day'] = range(end-start)
    rows = json.dumps(df.to_dict(orient='records'), indent=2)
    data = { 'rows': rows, 'cols': cols}
    return data


@app.route("/us_states_json")
def us_states_json():
    with open('us.json') as data_file:
        data = json.load(data_file)
    data = json.dumps(data)#, default=json_util.default)
    return data

@app.route("/get_time_series_data/<state>/<column>")
def time_series_data(state, column):
    isaggr = request.args.get('aggr', False)
    startDate = request.args.get('startDate', '01/01/2020') or '01/01/2020'
    endDate = request.args.get('endDate', '5/20/2020') or '5/20/2020'
    startDate = datetime.datetime.strptime(startDate, '%m/%d/%Y')
    endDate = datetime.datetime.strptime(endDate, '%m/%d/%Y')

    print(startDate, endDate)
    df_aggr = pd.read_csv('static/data/covid19_usa_complete.csv')
    if state!='' and state!='all':
        df_aggr = df_aggr[df_aggr.name == state]
    df_aggr = df_aggr.sort_values('Last Update')
    df_aggr['date'] = df_aggr['Last Update'].map(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
    datecheck = (df_aggr['date'] >= startDate) & (df_aggr['date'] <= endDate)
    df_aggr = df_aggr[datecheck]
    if not isaggr:
        return {"values": df_aggr[column].tolist(), "dates": df_aggr['Last Update'].map(lambda x: x[:10]).tolist()}
    return str(df_aggr[column].sum())


@app.route("/get_map_data")
def get_map_data():
    startDate = request.args.get('startDate', '01/01/2020') or '01/01/2020'
    endDate = request.args.get('endDate', '5/20/2020') or '5/20/2020'
    startDate = datetime.datetime.strptime(startDate, '%m/%d/%Y')
    endDate = datetime.datetime.strptime(endDate, '%m/%d/%Y')

    print(startDate, endDate, request.args.get('startDate'), request.args.get('endDate'))
    df_aggr = pd.read_csv('static/data/covid19_usa_complete.csv')
    df_aggr = df_aggr.rename(columns = {'name':'states'})
    # if state!='' and state!='all':
    #     df_aggr = df_aggr[df_aggr.states == state]
    df_aggr = df_aggr.sort_values('Last Update')
    df_aggr['date'] = df_aggr['Last Update'].map(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
    datecheck = (df_aggr['date'] >= startDate) & (df_aggr['date'] <= endDate)
    df_aggr = df_aggr[datecheck]
    df_aggr = df_aggr.groupby('states').agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Region': 'any'})
    # print(df_aggr)
    return df_aggr[['Confirmed', 'Deaths', 'Recovered']].to_csv(header=True)

if __name__ == "__main__":
    app.run('localhost', '5050')

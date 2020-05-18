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
import datetime

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
        if state == 'all':
            df_aggr =  df_aggr.groupby(['date']).sum()
            return {"values":df_aggr[column].tolist(), "dates": df_aggr.index.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d') ).tolist()}
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

import pandas as pd
import glob
import datetime
import os
from datetime import timedelta, date

fileprefix = 'nssac-ncov-sd-'
startdate = date(2020, 4, 14)
is_map_data = False
if is_map_data:
    df_aggr = pd.DataFrame()
    num_days = 31
    for i in range(num_days):
        cur = startdate + timedelta(days=i)
        filename = fileprefix + cur.strftime("%m-%d-%Y") + '.csv'
        filename = os.path.join('nssac-ncov-data-country-state', filename)
        if os.path.isfile(filename):
            # print(filename, 'yes')
            df = pd.read_csv(filename)
            df = df[df.Region == 'USA']
            df_aggr = pd.concat([df_aggr, df])
    df_aggr = df_aggr.groupby('name').agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum', 'Region': 'any'})
    print(df_aggr)
    df_aggr[['Confirmed', 'Deaths', 'Recovered']].to_csv('static/data/deaths_30days.csv', header=True)
    df_aggr['Deaths'].to_csv('static/data/deaths_only_30days.csv', header=True)
    print(len(df), df.columns, len(df_aggr), df_aggr.columns)

df_aggr = pd.DataFrame()
files = glob.glob('nssac-ncov-data-country-state/*')
for file_path in files[:]:
    df = pd.read_csv(file_path)
    df = df[df.name == 'New York']
    df_aggr = pd.concat([df_aggr, df])
print(df_aggr)
df_aggr
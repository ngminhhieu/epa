import requests
import csv
import os
import numpy
import pandas

URL = "https://aqs.epa.gov/data/api/sampleData/byCounty"
email = "hieu.nm151338@sis.hust.edu.vn"
key = "saffronbird49"
param = 88101

county_state = pandas.read_csv('location.csv')
county_state['County Code'] = county_state['County Code'].str.replace("County code: ", "")
county_state['State'] = county_state['State'].str.replace('State: ', '')

bdate = [20100101]
edate = [20100102]

for i in range(10):
    bdate.append(bdate[i] + 10000)
    edate.append(edate[i] + 10000)

coordinates = []
for index in range(len(county_state)):
    print(index)
    for index_time in range(len(bdate)):
        PARAMS = {
            'email': email,
            'key': key,
            'param': param,
            'bdate': bdate[index_time],
            'edate': edate[index_time],
            'state': county_state.iloc[index, 0],
            'county': county_state.iloc[index, 1]
        }

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        main_data = data['Data']
        if len(main_data) != 0:
            coordinates.append([main_data[0]['latitude'], main_data[0]['longitude']])
            break
    
df = pandas.DataFrame(coordinates, columns = ['Latitude', 'Longitude'])
result = pandas.concat([county_state, df], axis=1)
result.to_csv("result.csv")
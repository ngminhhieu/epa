
import requests
import csv
import os
import numpy
import pandas

URL = "https://aqs.epa.gov/data/api/sampleData/byCounty"
email = "hieu.nm151338@sis.hust.edu.vn"
key = "saffronbird49"
param = 88101
state = [x for x in range(1,321,2)]
county = 183

edate = []
bdate = []
for i in range(10):
    bdate.append(20100101 + 10000*i)
    edate.append(20101231 + 10000*i)

df = pandas.DataFrame(columns=['date_local', 'time_local', 'sample_measurement'])

for i in range(len(bdate)):
    print(i)
    PARAMS = {
    'email': email,
    'key': key,
    'param': param,
    'bdate': bdate[i],
    'edate': edate[i],
    'state': state,
    'county': county
    }

    r = requests.get(url=URL, params=PARAMS)
    data = r.json()
    main_data = data['Data']
    for j in range(len(main_data)):
        print(j)
        df = df.append({'date_local': main_data[j]['date_local'],
                        'time_local': main_data[j]['time_local'],
                        'sample_measurement': main_data[j]['sample_measurement']
                        }, ignore_index=True)

df.to_csv('test.csv')
import requests
import csv
import os
import numpy
import pandas

URL = "https://aqs.epa.gov/data/api/sampleData/byCounty"
email = "hieu.nm151338@sis.hust.edu.vn"
key = "saffronbird49"
param = 88101
county = []
numeric_county = [x for x in range(1, 321, 2)]
for i in range(len(numeric_county)):
    x = numeric_county[i]
    if x > 0 and x < 10:
        county.append("00" + str(x))
    elif x >= 10 and x < 100:
        county.append("0" + str(x))
    else:
        county.append(str(x))
state = 37

edate = []
bdate = []
for i in range(10):
    bdate.append(20100101 + 10000 * i)
    edate.append(20101231 + 10000 * i)

main_df = pandas.DataFrame(columns=['date_local', 'time_local'])

statistic = []

for index_county in range(len(county)):
    column_name = 'county_' + county[index_county]
    df = pandas.DataFrame(columns=[column_name])
    for index_date in range(len(bdate)):
        PARAMS = {
            'email': email,
            'key': key,
            'param': param,
            'bdate': bdate[index_date],
            'edate': edate[index_date],
            'state': state,
            'county': county[index_county]
        }

        r = requests.get(url=URL, params=PARAMS)
        data = r.json()
        main_data = data['Data']
        for index_data in range(len(main_data)):
            df = df.append(
                {column_name: main_data[index_data]['sample_measurement']},
                ignore_index=True)
        if (len(main_df) == 0):
            for index_data in range(len(main_data)):
                main_df = main_df.append(
                    {
                        'date_local': main_data[index_data]['date_local'],
                        'time_local': main_data[index_data]['time_local']
                    },
                    ignore_index=True)
        if (data['Header'][0]['rows'] != 0):
            with open('statistic.csv', 'a+') as result_file:

                statistic.append([
                    1, "Year: " + str(bdate[index_date]) + "\County code: " +
                    str(county[index_county]) + "\Rows: " +
                    str(data['Header'][0]['rows'])
                ])
                wr = csv.writer(result_file, dialect='excel')
                wr.writerows([statistic[-1]])
                result_file.close()

    if (len(main_df) > 0):
        if (len(df) != 0):
            main_df = pandas.concat([main_df, df], axis=1)
            main_df.to_csv('result.csv')
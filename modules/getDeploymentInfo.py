# def add(locationCode, dateFrom, dateTo):
#     download_path = '/tmp/ACMQAQC/data/' + str(locationCode) + '-' + str(dateFrom) + '-' + str(dateTo) + '/'
#     return download_path

# this code does a similar job of getting deployment information as the perl code that scraps
# #the web page "https://data.oceannetworks.ca/api/deployments"

import requests
import json
from onc.onc import ONC
import pandas as pd
import query_db
import datetime
import sys

with open('/home/ze/Documents/Credintials/myocean2.0token.csv') as token_file:
    my_token = token_file.read()

token = my_token.strip('\\n')



locationCode: str = sys.argv[1].upper()

for_process_and_plot_data_script = "/home/ze/Development_Related/api_downloads/onc_locationName_and_dataPath"

with open(for_process_and_plot_data_script, "w") as text_file:
    print(locationCode, file=text_file)
    text_file.close()
# def get_dep_info():

# check strings - don't need it anymore
def isitin(mysubstring, mystring):
    if (mysubstring in mystring) == True:
        return True
    else:
        return False


deviceCategoryCodes = query_db.query_db(
    "SELECT devicecategorycode FROM devicecategory  WHERE devicecategorycode  like '%ADCP%'  ORDER BY devicecategorycode DESC LIMIT 100")

# locationCode: str = input().upper()

# locationCode = 'NCBC' #'BACUS'  # for future get location codes from database  where there are adcp's

# Creating an empty Dataframe with column names only
dfObj = pd.DataFrame(columns=['locationCode', 'deviceCategoryCode', 'deviceCode', 'begin', 'end'])

for i in range(0, len(deviceCategoryCodes)):

    deviceCategoryCode = str(deviceCategoryCodes['devicecategorycode'][i])

    url = "https://data.oceannetworks.ca/api/deployments"
    parameters = {'method': 'get', 'token': token,
                  'deviceCategoryCode': deviceCategoryCode, 'locationCode': locationCode}
    deployments = []
    response = requests.get(url, params=parameters)
    k = -1

    if (response.ok):
        deps = json.loads(str(response.content, 'utf-8'))

    else:
        if (response.status_code == 400):
            error = json.loads(str(response.content, 'utf-8'))
            print(error)
        else:
            print('Error {} - {}'.format(response.status_code, response.reason))

    # print('LocationCode ', '  deviceCode', ' START  ', '               END', '                     Depth (m)',
    #       'deviceHeading (deg)')

    if len(deps) != 0:
        da = pd.DataFrame(deps)
        df = pd.DataFrame(da[['locationCode', 'deviceCode', 'begin', 'end']])
        #  print(" -------------------------\n")
        for idx in range(0, len(df['deviceCode'])):
            if isitin('RDIADCP', df.loc[0, 'deviceCode']) | isitin('RDI', df.loc[0, 'deviceCode']) == True:
                # print('LocationCode is ' + locationCode)
                # print('locationCode:' + locationCode + ':' + ' Found an RDI ADCP with device code ' + df.loc[idx, 'deviceCode'] +
                #       ' in ' + df.loc[idx, 'begin'] + ' deployment \n')
                # print(df)
                if df.loc[idx, 'end'] is None:
                    df.loc[idx, 'end'] = ONC.formatUtc(datetime.date.today())

                dfObj = dfObj.append({'locationCode': df.loc[idx, 'locationCode'],
                                      'deviceCategoryCode': deviceCategoryCode,
                                      'deviceCode': df.loc[idx, 'deviceCode'],
                                      'begin': df.loc[idx, 'begin'], 'end': df.loc[idx, 'end']}, ignore_index=True)

                download_path = "/home/ze/Development_Related/api_downloads/adcp/data/" + \
                                df.loc[idx, 'locationCode'] + '/' + df.loc[idx, 'locationCode'] + '-' + df.loc[idx, 'begin'] + '-' + df.loc[idx, 'end']
                with open(for_process_and_plot_data_script, "a") as text_file:
                    print(download_path, file=text_file)
                    text_file.close()

            else:
                print('\n', 'NO RDI ADCP in this deployment\n')

#    for j in range(0, len(deps)):  'locationCode', 'deviceCode', 'begin', 'end'
#    #if isitin('RDIADCP', deps[j]['deviceCode']) == True:
# #       print(deps[j]['locationCode'], deps[j]['deviceCode'], deps[j]['begin'], '', deps[j]['end'], '',
# #             deps[j]['depth'], '', deps[j]['heading'])
#        deps_pd = pd.DataFrame({'locationCode': deps[j]['locationCode'], 'deviceCode': deps[j]['deviceCode'],
#                                'begin': deps[j]['begin'], 'end': deps[j]['end']},index=range(0,len(deps)))
#        alldeps.append(deps_pd)


# print(type(dfObj))

file_name = '/home/ze/Development_Related/alladcpdeployments/' + locationCode + '_deployment_info'
dfObj.to_csv(file_name, encoding='utf-8', index=False)


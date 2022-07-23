from onc.onc import ONC
import requests
import json
import os
from contextlib import closing
import errno
import datetime
import dir_check

# -------------------------------------------------------------------------------------
# Eg., an example location code is RCNE3
print('READ this first!')
print('I will ask you to enter parameters relevant to this download. Provide those and hit return/enter key.')
print('No need to use quotation marks in your response.')
print('If you make mistakes, you can interrupt the kernel and re-enter values.')
print('There are shortcuts but are machine/browser dependent, so I leave them for you to figure out.')
print('\n', 'Location codes can be found at https://wiki.oceannetworks.ca/display/O2A/Available+Deployments')

print('\n', '************************************************************************')
print('Provide location code ')
print('Eg., an example location code is RCNE3')

locationCode: str = input().upper()

print('Enter deviceCategoryCode')
print('deviceCategoryCode (eg., CURRENTMETER)')
deviceCategoryCode: str = input()

print('\n', '************************************************************************')
print('Enter dateFrom (It cannot be outside of time device deployment period for the location ID)')
print('dateFrom (eg., 2016-07-27T00:00:00.000Z)')

dateFrom: str = input()

print('\n', '************************************************************************')

print('Enter dateTo (It cannot be outside of time device deployment period for the location ID)')
print('dateTo (eg., 2016-08-01T00:00:00.000Z)')
dateTo: str = input()

print('\n', '************************************************************************')

print("Enter acm for point current meters. Enter adcp for profile's")
acmorprofiler: str = input().lower()


def add(locationcode, datefrom, dateto):
    global path
    if acmorprofiler == 'acm':
        path = '/home/ze/Development_Related/api_downloads/acm/data/' + locationcode + '/' + locationcode + \
               '-' + datefrom + '-' + dateto + '/'
    elif acmorprofiler == 'adcp':
        path = '/home/ze/Development_Related/api_downloads/adcp/data/' + locationcode + '/' + locationcode + \
               '-' + datefrom + '-' + dateto + '/'
    return path


download_path = add(locationCode, dateFrom, dateTo)
# -------------------------------------------------------------------------------------
# for_process_and_plot_data_script = "/home/ze/Development_Related/api_downloads/onc_locationName_and_dataPath"
# # there will always be only 1 file with this name at a time
# with open(for_process_and_plot_data_script, "w") as text_file:
#     print(locationCode, file=text_file)
#     text_file.close()
# -------------------------------------------------------------------------------------
with open('/home/ze/Documents/Credintials/myocean2.0token.csv', "r") as token_file:
    my_token = token_file.read()
    token_file.close()

onc = ONC(my_token, outPath=download_path)

dateToFlag = 0
if len(dateTo) == 0:
    dateTo = ONC.formatUtc(datetime.date.today())
    dateToFlag = 1
    mydateTo = dateTo

myLogFileName = "/home/ze/Development_Related/api_downloads/myDataDownloadLog.log"
logTime: str = str(datetime.datetime.today())

if (acmorprofiler == 'acm') and (
        dir_check.dir_check(logTime, myLogFileName, download_path) == 1):  # 1 is fod download. 0 is not.
    print('\n', 'Thank you. your csv file(s) will be downloaded to ' + download_path)
    parameters = {'method': 'request',
                  'locationCode': locationCode,  # Barkley Canyon / Axis (POD 1)
                  'deviceCategoryCode': deviceCategoryCode,
                  # 'CURRENTMETER',  # 150 kHz Acoustic Doppler Current Profiler
                  'dataProductCode': 'TSSD',  # Time Series Scalar Data
                  'extension': 'csv',  # Comma Separated spreadsheet file
                  'dateFrom': dateFrom,  # The datetime of the first data point (From Date)
                  'dateTo': dateTo,  # The datetime of the last data point (To Date)
                  'dpo_qualityControl': 1,
                  # The Quality Control data product option - See https://wiki.oceannetworks.ca/display/DP/1
                  'dpo_resample': 'none',
                  # The Resampling data product option - See https://wiki.oceannetworks.ca/display/DP/1
                  'dpo_dataGaps': 0}
elif (acmorprofiler == 'adcp') and (dir_check.dir_check(logTime, myLogFileName, download_path) == 1):  # need to
    # handle the dir_check () = 0 case properly . It could be done better from within the shell scripts though.
    print('\n', 'Thank you. your .nc file(s) will be downloaded to ' + download_path)
    parameters = {'locationCode': locationCode,
                  'deviceCategoryCode': deviceCategoryCode,
                  'dataProductCode': "RADCPTS",
                  'extension': 'nc',
                  'dateFrom': dateFrom,
                  'dateTo': dateTo,
                  'dpo_3beam': 'Off',
                  'dpo_errVelScreen': 0,
                  'dpo_falseTarScreen': 255,
                  'dpo_corScreen': 0,
                  'dpo_ensemblePeriod': 3600,
                  'dpo_velocityBinmapping': 0}
    try:
        results = onc.orderDataProduct(parameters, includeMetadataFile=False)
        downloadResults = results["downloadResults"]
        first_key = list(downloadResults)[0]
        filePath = download_path + first_key['file']
        print('\n', "Your file is ready at path '{}'".format(filePath))
    except NameError:
        pass

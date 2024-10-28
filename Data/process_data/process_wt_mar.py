#Closely adapted from the Decarbonization-Potential project    https://github.com/umassos/decarbonization-potential/tree/master/

import pandas as pd
import json
import os
import sys

interval = 5
if len(sys.argv) > 1:
    interval = int(sys.argv[1])
    if interval % 5 != 0:
        raise ValueError("Raw WattTime Data is in five minute intervals. Input interval must be a multiple of five.")
rows_in_inteval = int(interval/5)

zone_dict = {
    "SGP": "SG",
    "IE": "IE",
    "NEM-NSW": "AU-NSW",
    "JP-TK": "JP-TK",
    "KOR": "KR",
    "FR": "FR",
    "DE": "DE",
    "IT": "IT-NO",
    "UK": "GB",
    "BRA": "BR-CS",
    "PJM-DC": "US-MIDA-PJM",
    "NL": "NL",
    "MISO-MASON-CITY": "US-MIDW-MISO",
    "ERCOT-NORTHCENTRAL": "US-TEX-ERCO",
    "JP-KN": "JP-KN",
    "IND": "IN-WE",
    "CAISO-NORTH": "US-CAL-CISO",
    "BPA": "US-NW-BPAT",
    "PJM-EASTERN-OH": "US-MIDA-PJM",
    "IESO-SOUTH": "CA-ON",
    "PL": "PL",
    "SE": "SE",
    "NEM-VIC": "AU-VIC",
    "ES": "ES",
    }

sourceDir = '../WattTime'
postProcessDir = '../post_processing'
time_range = '2022-01-01-2022-01-31'
raw = os.listdir(sourceDir)

zone_code_list = []
for file in raw:
    zone_code, time_range = file[:-5].split('_')
    zone_code_list.append(zone_code)

combined_df = pd.DataFrame()
first = True
for zone_code in zone_code_list:
    zone_temp_df = pd.DataFrame()
    zone_file = os.path.join(sourceDir, f'{zone_code}_{time_range}.json')
    data = json.load(open(zone_file))
    df = pd.DataFrame(data["data"])
    raw_df = df.iloc[::rows_in_inteval]
    raw_df.loc[:, 'value'] *=(453.592/1000)     #convert pounds/megawatts hours to grams/kilowatts hours
    raw_df.reset_index(drop=True, inplace=True)

    carbon_trace = raw_df['value']
    zone_temp_df = pd.concat([zone_temp_df, carbon_trace])

    zone_name = zone_dict[zone_code]
    zone_temp_df.columns = [zone_name]
    zone_temp_df.reset_index(drop=True, inplace=True)

    if first:
        first = False
        combined_df = pd.to_datetime(raw_df['point_time'].str.replace('00+00:', '').str.replace(' ', ''))
    combined_df  = pd.concat([combined_df,zone_temp_df], axis=1)

combined_df.rename(columns={'point_time': 'datetime'}, inplace=True)

save_name = os.path.join(postProcessDir, 'watttime_ci.csv')
combined_df.to_csv(save_name, index=False)
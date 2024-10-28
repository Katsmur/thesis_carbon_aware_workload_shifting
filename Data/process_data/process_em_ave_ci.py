#Closely adapted from the Decarbonization-Potential project    https://github.com/umassos/decarbonization-potential/tree/master/

import pandas as pd
import datetime
import os
import sys

sourceDir = '../average_carbon_intensity'
postProcessDir = '../post_processing'
raw = os.listdir(sourceDir)

carbon_col = 'Carbon Intensity gCO₂eq/kWh (LCA)'
if len(sys.argv) > 1:
    if sys.argv[1] == 'direct':
        carbon_col = 'Carbon Intensity gCO₂eq/kWh (direct)'
    elif sys.argv[1] != 'LCA':
        raise ValueError(f"Unsupported commandline argument {sys.argv[1]}")

year_dict = dict()

for file in raw:
    zone_code, year, granular = file[:-4].split('_')
    if year not in year_dict:
        year_dict[year] = []
    year_dict[year].append(zone_code)

year_list = list(year_dict.keys())
year_list.sort()

zone_code_list = year_dict[year_list[0]]
l = len(zone_code_list)
file_check = all(len(year_dict[y]) == l for y in year_dict)

if not file_check: 
    print("Not all zone codes have the same number of years!")
    exit()

start_year = year_list[0]
end_year = year_list[-1]
start_date = datetime.datetime.strptime(f'{start_year}-01-01', "%Y-%m-%d")
end_date = datetime.datetime.strptime(f'{int(end_year)+1}-01-01', "%Y-%m-%d")
datelist = pd.date_range(start_date, end_date, freq="H", inclusive='left')

combined_df = pd.DataFrame()
combined_df['datetime'] = datelist

for zone_code in zone_code_list:

    zone_temp_df = pd.DataFrame()
    for year in year_list: 

        zone_year_file = os.path.join(sourceDir, f'{zone_code}_{year}_hourly.csv')

        raw_df = pd.read_csv(zone_year_file)
        carbon_trace = raw_df[carbon_col]

        zone_temp_df = pd.concat([zone_temp_df, carbon_trace])
    
    zone_temp_df.columns = [zone_code]
    zone_temp_df.reset_index(drop=True, inplace=True)

    combined_df  = pd.concat([combined_df,zone_temp_df], axis=1)

#combined_df[zone_code_list] = combined_df[zone_code_list].bfill().ffill()


save_name = os.path.join(postProcessDir, 'average_ci.csv')
combined_df.to_csv(save_name, index=False)
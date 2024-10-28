#Closely adapted from the Decarbonization-Potential project    https://github.com/umassos/decarbonization-potential/tree/master/

import pandas as pd
import os

sourceDir = '../marginal_carbon_intensity'
postProcessDir = '../post_processing'
raw = os.listdir(sourceDir)

zone_code_list = []
for file in raw:
    marginal, emissions, zone_code = file[:-4].split('_')
    zone_code_list.append(zone_code)

combined_df = pd.DataFrame()
first = True
for zone_code in zone_code_list:
    zone_temp_df = pd.DataFrame()
    zone_file = os.path.join(sourceDir, f'marginal_emissions_{zone_code}.csv')
    raw_df = pd.read_csv(zone_file)
    carbon_trace = raw_df['marginal_carbon_intensity_avg']
    zone_temp_df = pd.concat([zone_temp_df, carbon_trace])

    zone_temp_df.columns = [zone_code]
    zone_temp_df.reset_index(drop=True, inplace=True)

    if first:
        first = False
        combined_df = raw_df['target_time']
    combined_df  = pd.concat([combined_df,zone_temp_df], axis=1)

combined_df['target_time'] = combined_df['target_time'].str.replace('+00:00', '')
combined_df.rename(columns={'target_time': 'datetime'}, inplace=True)

save_name = os.path.join(postProcessDir, 'marginal_ci.csv')
combined_df.to_csv(save_name, index=False)
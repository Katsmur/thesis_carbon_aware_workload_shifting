import pandas as pd
from datetime import timedelta
import os
import sys

interval = 60
if len(sys.argv) > 1:
    interval = int(sys.argv[1])

sourceDir = '../LWA_data'
postProcessDir = '../post_processing'
raw = os.listdir(sourceDir)

zone_code_list = []
for file in raw:
    zone_code, _ = file[:-4].split('_')
    zone_code_list.append(zone_code)

combined_df = pd.DataFrame()
for zone_code in zone_code_list:
    zone_file = os.path.join(sourceDir, f'{zone_code}_ci.csv')
    raw_df = pd.read_csv(zone_file, index_col=0, parse_dates=True)
    
    raw_df = raw_df.asfreq(freq=timedelta(minutes=int(interval)), method='ffill')
    raw_df.columns = [zone_code]
    
    combined_df  = pd.concat([combined_df,raw_df], axis=1)

combined_df.reset_index(inplace=True)
combined_df.rename(columns={'Time': 'datetime'}, inplace=True)

save_name = os.path.join(postProcessDir, 'LWA_ci.csv')
combined_df.to_csv(save_name, index=False)
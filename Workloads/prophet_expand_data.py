import os
import sys
import pandas as pd
import datetime
currdir = os.path.dirname(__file__)
azure_module = os.path.join(currdir,"./",  "Azure")
sys.path.append(azure_module)
import Azure_processing
from prophet import Prophet

extension_period = 336

def azure_2021_extension(extension_period):
    data_df = pd.read_csv('./Azure/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt', usecols=[2,3])
    data_df = Azure_processing.AzureFunctionsInvocationTraceForTwoWeeksJan2021_proc(data_df)
    last_hour = data_df['start_hour'].iloc[-1]

    df = data_df.rename(columns={'start_hour': "ds", 'count': "y"})
    df['ds'] = pd.to_datetime(df.ds*3600, origin=pd.Timestamp('2021-01-01'), unit='s')

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=extension_period, freq = 'H')
    forecast = m.predict(future)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    x = forecast['yhat_upper'].iloc[extension_period-1:-1].astype(int).abs().tolist()
    y = data_df['count'].tolist()
    y = y + x
    df = pd.DataFrame(y, columns=['jobs'])
    df.insert(loc=0, column='start_hour', value=df.index)

    postProcessDir = './Azure'
    save_name = os.path.join(postProcessDir, 'expanded_Azure_2021.csv')
    df.to_csv(save_name, index=False)

def azure_2019_extension(extension_period):
    x = Azure_processing.AzureFunctions2019_total_workload('./Azure/AzureFunctionsDataset2019')
    totaldf = pd.DataFrame(x, columns =['ds', 'y'])

    m = Prophet()
    m.fit(totaldf)
    future = m.make_future_dataframe(periods=extension_period, freq = 'H')
    forecast = m.predict(future)

    df = pd.DataFrame(forecast['yhat'], columns=['jobs'])
    df.insert(loc=0, column='start_hour', value=df.index)

    postProcessDir = './Azure/AzureFunctionsDataset2019'
    save_name = os.path.join(postProcessDir, 'expanded_Azure_2019.csv')
    df.to_csv(save_name, index=False)

azure_2021_extension(extension_period)
print('Saved Expanded Azure Workload')
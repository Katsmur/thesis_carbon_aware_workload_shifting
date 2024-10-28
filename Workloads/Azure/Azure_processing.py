import pandas as pd
import os
import sys
from datetime import datetime, timedelta
currdir = os.path.dirname(__file__)
workload_module = os.path.join(currdir,"../../",  "Workloads")
sys.path.append(workload_module)
import workload_API


def AzureFunctionsInvocationTraceForTwoWeeksJan2021():
    toptwoweeks = [
        'bd5be891d0d10fbc3c59215d5f8159ea496433bc41adba7d8d10ea21d35c3e3a',
        '4753d9ffc8a362eadfb27498581969561c0ddbec7257759dbcb9936e506aff82',
        'cc5bb2108cc7daf53f9728ad21f661a8ef9c8b36284bacfcb712e2be87eef842',
        '8ef9c2c907e9cd5fb4b5cb3bfbebca6d9a01bb1f65a0f476e3914cae25bd2aae',
        '155e47f8e7f751d0c845049456d01832013c61336a8cd85901330ac821a71534'
    ]
    x = read_AzureFunctionsInvocationTraceForTwoWeeksJan2021()
    selected_func_df = x.loc[x['func'].isin(toptwoweeks)].drop(columns=['func'])
    
    return AzureFunctions2021API(selected_func_df)#AzureFunctionsInvocationTraceForTwoWeeksJan2021_proc(selected_func_df)


def read_AzureFunctionsInvocationTraceForTwoWeeksJan2021():
    return pd.read_csv('./Workloads/Azure/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt', usecols=[1,2,3])
    

def AzureFunctions2021API(azureinvoDF):
    x = azureinvoDF['end_timestamp'] - azureinvoDF['duration']
    azureinvoDF['start_timestamp'] = pd.to_datetime(x.astype(int), origin=pd.Timestamp('2021-01-31'), unit='s')#.astype(str)
    new_df = pd.DataFrame({'num_of_requests' : azureinvoDF.groupby(['start_timestamp', 'duration']).size().astype(int)}).reset_index()
    new_df['location'] = pd.Series(['N/A' for _ in range(len(azureinvoDF.index))])
    
    temp = new_df[['start_timestamp', 'duration', 'location', 'num_of_requests']]
    api_object = workload_API.Job_DF(temp)
    return api_object


def AzureFunctionsInvocationTraceForTwoWeeksJan2021_proc(azureinvoDF):
    x = (azureinvoDF['end_timestamp'] - azureinvoDF['duration']) / 3600
    azureinvoDF['start_hour'] = x.astype(int)
    requests_per_hour = azureinvoDF.start_hour.value_counts().reindex(range(azureinvoDF.start_hour.iloc[-1]+1), fill_value=0)
    req_df = requests_per_hour.to_frame()
    req_df = req_df.reset_index()
    req_df.start_hour = pd.to_datetime(req_df.start_hour*3600, origin=pd.Timestamp('2021-01-31'), unit='s')
    
    return req_df

def AzureFunctions2019_Top_Funcs():
    azure1 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d01.csv')
    azure1['total'] = azure1[azure1.columns[4:]].sum(axis=1)

    azure2 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d02.csv')
    azure2['total'] = azure2[azure2.columns[4:]].sum(axis=1)

    azure3 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d03.csv')
    azure3['total'] = azure3[azure3.columns[4:]].sum(axis=1)

    azure4 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d04.csv')
    azure4['total'] = azure4[azure4.columns[4:]].sum(axis=1)

    azure5 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d05.csv')
    azure5['total'] = azure5[azure5.columns[4:]].sum(axis=1)

    azure6 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d06.csv')
    azure6['total'] = azure6[azure6.columns[4:]].sum(axis=1)

    azure7 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d07.csv')
    azure7['total'] = azure7[azure7.columns[4:]].sum(axis=1)

    azure8 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d08.csv')
    azure8['total'] = azure8[azure8.columns[4:]].sum(axis=1)

    azure9 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d09.csv')
    azure9['total'] = azure9[azure9.columns[4:]].sum(axis=1)

    azure10 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d10.csv')
    azure10['total'] = azure10[azure10.columns[4:]].sum(axis=1)

    azure11 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d11.csv')
    azure11['total'] = azure11[azure11.columns[4:]].sum(axis=1)

    azure12 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d12.csv')
    azure12['total'] = azure12[azure12.columns[4:]].sum(axis=1)

    azure13 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d13.csv')
    azure13['total'] = azure13[azure13.columns[4:]].sum(axis=1)

    azure14 = pd.read_csv('./Workloads/Azure/AzureFunctionsDataset2019/invocations_per_function_md.anon.d14.csv')
    azure14['total'] = azure14[azure14.columns[4:]].sum(axis=1)

    frames = [azure1[['HashFunction', 'total']], azure2[['HashFunction', 'total']], azure3[['HashFunction', 'total']], azure4[['HashFunction', 'total']], azure5[['HashFunction', 'total']], azure6[['HashFunction', 'total']], azure7[['HashFunction', 'total']], azure8[['HashFunction', 'total']], azure9[['HashFunction', 'total']], azure10[['HashFunction', 'total']], azure11[['HashFunction', 'total']], azure12[['HashFunction', 'total']], azure13[['HashFunction', 'total']], azure14[['HashFunction', 'total']]]
    result = pd.concat(frames)
    sumdf = result.groupby('HashFunction').sum()
    sumdf = sumdf.sort_values(by=['total'])

    save_name = os.path.join('./Workloads/Azure/AzureFunctionsDataset2019/', 'output_2019_function_count.csv')
    sumdf.to_csv(save_name, header=False, index=False)
    sumdf.tail()

def AzureFunctions2019_workload_per_day(day, file_loc):
    top_funcs = [
        '063ba6f3c1d425f6f5c3bde3b9ba1eba7c6d81c57fd794860338638212d30dfb',
        '93d4c31373200d74272af6e0feb443ea1206b83034d563ad7bc934e89b12e170',
        'dd833bb70b3a57caaa6b4e4560975d9ba5a77fd0151e13d1b5f30b4f6c381d5c',
        '5315be05fc3b21a3f483ed0759bce825764dcf8a762623a1d94ff63f9d9ce4cc',
        '8203ff88388384a6f9ed28664e8e9484119ff340cb7dc0811a15194b3a507f0e'
    ]
    
    fileString = f'{file_loc}/invocations_per_function_md.anon.d' + day + '.csv'
    azuredf = pd.read_csv(fileString)

    sel_func_series = azuredf.loc[azuredf['HashFunction'].isin(top_funcs)][azuredf.columns[4:]].sum()
    #sel_hourly = sel_func_series.groupby((sel_func_series.index.astype(int) - 1) // 60).sum()

    return sel_func_series

def AzureFunctions2019_duration_per_day(day, file_loc):
    top_funcs = [
        '063ba6f3c1d425f6f5c3bde3b9ba1eba7c6d81c57fd794860338638212d30dfb',
        '93d4c31373200d74272af6e0feb443ea1206b83034d563ad7bc934e89b12e170',
        'dd833bb70b3a57caaa6b4e4560975d9ba5a77fd0151e13d1b5f30b4f6c381d5c',
        '5315be05fc3b21a3f483ed0759bce825764dcf8a762623a1d94ff63f9d9ce4cc',
        '8203ff88388384a6f9ed28664e8e9484119ff340cb7dc0811a15194b3a507f0e'
    ]
    
    fileString = f'{file_loc}/function_durations_percentiles.anon.d' + day + '.csv'
    azuredf = pd.read_csv(fileString, usecols=[2,3,4])

    sel_func_dur = azuredf.loc[azuredf['HashFunction'].isin(top_funcs)]
    sel_func_dur.loc[:, 'Average'] *= 0.001
    sum = sel_func_dur.Count.sum()
    sel_func_dur.loc[:, 'Count'] /= sum

    return sel_func_dur[['Average', 'Count']].values.tolist()

def AzureFunctions2019_total_workload(file_loc = './Workloads/Azure/AzureFunctionsDataset2019'):
    date_string = '2019-01-'
    outputList = []

    for i in range(1,15):
        day_num = str(i).zfill(2)
        day_string = date_string + day_num
        datetime_object = datetime.strptime(day_string, '%Y-%m-%d')
        #datetime_object = datetime(date_object.year, date_object.month, date_object.day)
        duration_list = AzureFunctions2019_duration_per_day(day_num, file_loc)
        minuteSeries = AzureFunctions2019_workload_per_day(day_num, file_loc)
        for index, value in minuteSeries.items():
            datetime_ind = datetime_object + timedelta(minutes=int(index)-1)
            #datetime_string = datetime.strftime(datetime_ind, '%Y-%m-%d %H:%M:%S')
            for j in duration_list:
                outputList.append([datetime_ind, j[0], 'N/A', int(value*j[1])])
        
    api_object = workload_API.Job_DF(outputList, columns=['start_timestamp', 'duration', 'location', 'num_of_requests'])
    return api_object


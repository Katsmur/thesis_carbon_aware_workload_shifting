import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
currdir = os.path.dirname(__file__)

genai_module = os.path.join(currdir,"./Workloads/",  "GenerativeAI")
sys.path.append(genai_module)
import genAI

temp_module = os.path.join(currdir,"./Workloads/",  "Temporal")
sys.path.append(temp_module)
import temporalWorkload

carbon_min_em_module = os.path.join(currdir,"./Algorithms_Workload_Shifting/",  "Carbon_Min")
sys.path.append(carbon_min_em_module)
import carbon_min_algorithm

lets_wait_awhile_em_module = os.path.join(currdir,"./Algorithms_Workload_Shifting/",  "Lets_Wait_Awhile")
sys.path.append(lets_wait_awhile_em_module)
import LWA_algorithm

azure_module = os.path.join(currdir,"./Workloads/",  "Azure")
sys.path.append(azure_module)
import Azure_processing

net_module = os.path.join(currdir,"./",  "Cloud_Network_Zones")
sys.path.append(net_module)
import network_handler



#carbon_string = 'watttime'
#carbon_string = 'marginal'
carbon_string = 'average'
#carbon_string = 'LWA'

full_carbon_string = f'./Data/post_processing/{carbon_string}_ci.csv'
carbonDF = pd.read_csv(full_carbon_string, parse_dates=True)

def lwa_periodic_handle(origin, error, measurement_interval, jobs, job_type, ci_dataframe, ci_type, max_steps_window=False, experiment_name=False):
    if 'datetime' in ci_dataframe.columns:
        ci_dataframe['datetime'] = pd.to_datetime(ci_dataframe['datetime'], format='%Y-%m-%d %H:%M:%S')
        ci_df = ci_dataframe.set_index(['datetime'])
    else:
        ci_df = ci_dataframe

    ci_df = ci_df[origin:]

    ci_df = interval_selecter(ci_df, measurement_interval)

    if max_steps_window:
        result = LWA_algorithm.periodic_experiment(error=error, measurement_interval=measurement_interval, jobs=jobs, ci_dataframe=ci_df, max_steps_window=max_steps_window)
    else:
        result = LWA_algorithm.periodic_experiment(error=error, measurement_interval=measurement_interval, jobs=jobs, ci_dataframe=ci_df)
    
    if not os.path.exists(f'./output/lets_wait_awhile/{job_type}/{ci_type}'):
        os.makedirs(f'./output/lets_wait_awhile/{job_type}/{ci_type}')
    if experiment_name:
        save_name = os.path.join(f'./output/lets_wait_awhile/{job_type}/{ci_type}/', f'periodic_{experiment_name}.csv')
    else:
        save_name = os.path.join(f'./output/lets_wait_awhile/{job_type}/{ci_type}/', f'periodic_{error}.csv')
    pd.DataFrame(result).to_csv(save_name, index=False)


def lwa_adhoc_handle(origin, measurement_interval, ci_dataframe, ci_type: str, ml_jobs, job_type: str, forecast_method, interruptible: bool=False, error: float=False, specific_region: str=False, experiment_name=False):
    if 'datetime' in ci_dataframe.columns:
        ci_dataframe['datetime'] = pd.to_datetime(ci_dataframe['datetime'], format='%Y-%m-%d %H:%M:%S')
        ci_df = ci_dataframe.set_index(['datetime'])
    else:
        ci_df = ci_dataframe
        
    ci_df = ci_df[origin:]

    ci_df = interval_selecter(ci_df, measurement_interval)

    if specific_region:
        countries = [specific_region]
    else:
        countries = ci_dataframe.columns
    
    for country in countries:
        timeline, ci = LWA_algorithm.ml_experiment(measurement_interval=measurement_interval,ci_dataframe=ci_dataframe, ml_jobs=ml_jobs, forecast_method=forecast_method, interruptible=interruptible, error=error, specific_region=country)
    
        # Store results
        df = pd.DataFrame({"active_jobs": timeline, "ci": ci, "emissions": ci * timeline}, index=ci.index)
        i = "_i" if interruptible else ""
        e = f"_{error}" if error else ""
        exp = f"_{experiment_name}" if experiment_name else ""
        if not os.path.exists(f'./output/lets_wait_awhile/{job_type}/{ci_type}'):
            os.makedirs(f'./output/lets_wait_awhile/{job_type}/{ci_type}')
        save_name = os.path.join(f'./output/lets_wait_awhile/{job_type}/{ci_type}/', f'ml_{forecast_method}{e}{i}_{country}{exp}.csv')
        df.to_csv(save_name)


def carbonMin_output_formatter(request_df, zone_df,job_type, ci_type=carbon_string, experiment_name=False):
    name = f"_{experiment_name}" if experiment_name else ""
    if not os.path.exists(f'./output/carbon_min/{job_type}/{ci_type}'):
        os.makedirs(f'./output/carbon_min/{job_type}/{ci_type}')
    output_df = pd.DataFrame(request_df, columns=['datetime', 'requests', 'emissions'])
    save_name = os.path.join(f'./output/carbon_min/{job_type}/{ci_type}/', f'requests{name}.csv')
    output_df.to_csv(save_name, index=False)
    output_df = pd.DataFrame(zone_df)
    save_name = os.path.join(f'./output/carbon_min/{job_type}/{ci_type}/', f'zones_used{name}.csv')
    output_df.to_csv(save_name, header=False, index=False)


def position_dependent_workload(workload, interval, carbon_index=False, carbon_df=pd.DataFrame()):
    outputReqs = []
    outputZones = []
    carbonSum = 0
    
    if interval == 0:
        interval = 60

    if carbon_df.empty:
        global carbonDF
    else:
        carbonDF = carbon_df

    ci_datetime = carbonDF['datetime']
    carbonDFCopy = carbonDF.drop('datetime', axis=1)
    if carbon_index:
        carbonDFCopy = carbonDFCopy[carbon_index:].reset_index(drop=True)
        ci_datetime = ci_datetime[carbon_index:].reset_index(drop=True)
        

    carbon_min_algorithm.carbonMin_verifier(workload, interval)
    workload = carbon_min_algorithm.carbonMin_API_conversion(workload, interval)

    for index, y in enumerate(workload):
        carbHour = carbonDFCopy.iloc[index].sort_values()
        opEm = carbon_min_algorithm.carb_min_method(carbHour, y[1])
        #print(f"{index} datetime:{ci_datetime.iloc[index]}, requests at hour:{y[1]}, operation emissions at hour:{opEm[0] + opEm[1]}")
        outputReqs.append([ci_datetime.iloc[index], y[1], opEm[0] + opEm[1]])
        outputZones.append([ci_datetime.iloc[index]] + opEm[2:])
        carbonSum += (opEm[0] + opEm[1])
 
    return carbonSum, outputReqs, outputZones


#old
def datetime_dependent_workload(workload):
    outputReqs = []
    outputZones = []
    carbonSum = 0

    global carbonDF
    carbonDFCopy = carbonDF.set_index(['datetime'])

    for y in workload:
        carbHour = carbonDFCopy.loc[y[0]].sort_values()
        opEm = carbon_min_algorithm.carb_min_method(carbHour, y[1])
        print(f"datetime:{y[0]}, requests at hour:{y[1]}, operation emissions at hour:{opEm[0] + opEm[1]}")
        outputReqs.append([y[0], y[1], opEm[0] + opEm[1]])
        outputZones.append([y[0]] + opEm[2:])
        carbonSum += opEm[0] + opEm[1]
    return carbonSum, outputReqs, outputZones


def ci_len_checker(start_index=0, start_date=None, end_index=-1, end_date=None, carbon_df=pd.DataFrame()):
    global full_carbon_string
    end = end_index
    start = start_index

    if carbon_df.empty:
        global carbonDF
    else:
        carbonDF = carbon_df

    if end_date != None:
        end = carbonDF.index[carbonDF['datetime']==end_date].tolist()[0]
    elif end < 0:
        #raise ValueError("An end_index or end_date must be provided")
        end = len(carbonDF.index)
    
    if start_date != None:
        start = carbonDF.index[carbonDF['datetime']==start_date].tolist()[0]
    
    if (len(carbonDF.index)-start) < (end-start):
        raise ValueError(f"Carbon Intensity File smaller than indices {full_carbon_string}")
    return


def interval_selecter(carbon_df, interval):
    if carbon_df.index[1].minute - carbon_df.index[0].minute > interval:
        raise ValueError(f"Carbon Intensity data does not support an interval as small as {interval}")
    return carbon_df[carbon_df.index.minute % interval == 0]


def genAI_workload(interval, reqs_per_day=80825, start_day=datetime.strptime('2022-01-01', '%Y-%m-%d'), total_hours=744, countryDF=pd.DataFrame(), carbon_df=pd.DataFrame(), experiment_name=False):

    ci_len_checker(end_index=total_hours, carbon_df=carbon_df)
    carbon_df = interval_selecter(carbon_df, interval)
    
    #z_date = datetime.strptime(start_day, '%Y-%m-%d').date()
    temp_workload = genAI.workloadOverADay(requestsADay=reqs_per_day, total_hours=total_hours, start_date=start_day, countryDF=countryDF)
    temp_workload = temp_workload.drop(columns=['location', 'duration'])
    #temp_workload['start_timestamp'] = pd.to_datetime(temp_workload.start_timestamp)
    carbonSum, outputReqs, outputZones = position_dependent_workload(temp_workload, interval, carbon_df=carbon_df)

    carbonMin_output_formatter(outputReqs, outputZones,job_type='gen_AI', ci_type=carbon_string)
    """
    name = f"_{experiment_name}" if experiment_name else ""
    if not os.path.exists('./output/carbon_min/gen_AI/'):
        os.makedirs('./output/carbon_min/gen_AI/')
    output_df = pd.DataFrame(outputReqs)
    save_name = os.path.join('./output/carbon_min/gen_AI/', f'requests_{carbon_string}{name}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    output_df = pd.DataFrame(outputZones)
    save_name = os.path.join('./output/carbon_min/gen_AI/', f'zones_used_{carbon_string}{name}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    """
    return carbonSum


#old
def Azure_invocation_2021(flag=1):
    temp_workload = []
    if 0 == flag:
        temp_workloadDF = pd.read_csv('./Workloads/Azure/updated_Azure.csv')
        temp_workload = temp_workloadDF.values.tolist()
    elif 1 == flag:
        temp_workload = Azure_processing.AzureFunctionsInvocationTraceForTwoWeeksJan2021()
    
    x = carbon_min_algorithm.carbonMin_API_conversion(temp_workload)
    
    ci_len_checker(end_index=len(x))

    carbonSum, outputReqs, outputZones = position_dependent_workload(x)

    output_df = pd.DataFrame(outputReqs)
    save_name = os.path.join('./output/carbon_min/azure_2021_data/', f'requests_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    output_df = pd.DataFrame(outputZones)
    save_name = os.path.join('./output/carbon_min/azure_2021_data/', f'zones_used_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    return carbonSum


#old
def Azure_Functions_2019():
    x = Azure_processing.AzureFunctions2019_total_workload()
    carbonSum, outputReqs, outputZones = position_dependent_workload(x)

    output_df = pd.DataFrame(outputReqs)
    save_name = os.path.join('./output/carbon_min/azure_2019_data/', f'requests_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    output_df = pd.DataFrame(outputZones)
    save_name = os.path.join('./output/carbon_min/azure_2019_data/', f'zones_used_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    return carbonSum


def temporal_workload(interval, reqs_per_day=80825, start_index=0, end_index=744):
    ci_len_checker(start_index=start_index,end_index=end_index)
    
    temp_workload = temporalWorkload.t_workload(jobs = reqs_per_day, start_dt = start_index, end_dt = end_index, loc="FR", measurement_interval = interval)
    temp_workload = temp_workload.drop(columns=['location', 'duration'])

    carbonSum, outputReqs, outputZones = position_dependent_workload(temp_workload, interval, start_index) #datetime_dependent_workload(x)

    if not os.path.exists('./output/carbon_min/temporal/'):
        os.makedirs('./output/carbon_min/temporal/')
    output_df = pd.DataFrame(outputReqs)
    save_name = os.path.join('./output/carbon_min/temporal/', f'requests_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    output_df = pd.DataFrame(outputZones)
    save_name = os.path.join('./output/carbon_min/temporal/', f'zones_used_{carbon_string}.csv')
    output_df.to_csv(save_name, header=False, index=False)
    return carbonSum
    

def lwa_adhoc():
    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')  #temp_workload['start_timestamp'].iloc[0]
    interval = 30

    temp_workload = temporalWorkload.t_workload(jobs = 3387, measurement_interval=interval, start_dt = origin, end_dt = datetime.strptime("2020-12-31", '%Y-%m-%d'))
    temp_workload.loc[:, 'start_timestamp'] -= origin
    
    y = temp_workload.drop(columns=['location'])
    carbonDF['datetime'] = pd.to_datetime(carbonDF['datetime'], format='%Y-%m-%d %H:%M:%S')
    ci_df = carbonDF.set_index(['datetime'])
    ci_df = ci_df[origin:]

    interr = [True, False]
    errors = [0, 0.05, 0.1]
    forcastmethods = ["next_workday", "semi_weekly"]
    for i in interr:
        for error in errors:
            for forecast_method in forcastmethods:
                lwa_adhoc_handle(origin=origin, measurement_interval=interval,ci_dataframe=ci_df, ci_type=carbon_string, ml_jobs=y, job_type='temporal', forecast_method=forecast_method, interruptible=i, error=error)
    lwa_adhoc_handle(origin=origin,measurement_interval=interval,ci_dataframe=ci_df, ci_type=carbon_string, ml_jobs=y, job_type='temporal', forecast_method=0, interruptible=False, error=False)
    return


def lwa_periodic():
    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')  #temp_workload['start_timestamp'].iloc[0]
    interval = 30
    
    periodic_workload = temporalWorkload.periodic_temporal_workload(dur=interval, ori=origin)
    periodic_workload.loc[:, 'start_timestamp'] -= origin

    y = periodic_workload.drop(columns=['location', 'duration'])

    errors = [0, 0.05]
    for error in errors:
        lwa_periodic_handle(origin=origin, error=error, measurement_interval=interval, jobs=y, job_type='temporal', ci_type=carbon_string, ci_dataframe=carbonDF)
    return


def demo_periodic_2():
    ci_str = ['average', 'marginal']

    origin = datetime.strptime("2023-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    interval = 60

    periodic_workload = temporalWorkload.periodic_temporal_workload(dur=interval, ori=origin)
    periodic_workload.loc[:, 'start_timestamp'] -= origin
    periodic_workload = periodic_workload.drop(columns=['location', 'duration'])

    for cstr in ci_str:
        full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
        ci_df = pd.read_csv(full_carbon_string, parse_dates=True)[['datetime','FR','US-CAL-CISO','DE','GB']]

        lwa_periodic_handle(origin=origin, error=0, measurement_interval=interval, jobs=periodic_workload, job_type='temporal', ci_type=cstr, ci_dataframe=ci_df, experiment_name='demo_2')
    
    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    periodic_workload = temporalWorkload.periodic_temporal_workload(dur=interval, ori=origin)
    periodic_workload.loc[:, 'start_timestamp'] -= origin
    periodic_workload = periodic_workload.drop(columns=['location', 'duration'])

    full_carbon_string = './Data/post_processing/LWA_ci.csv'
    ci_df = pd.read_csv(full_carbon_string, parse_dates=True)
    lwa_periodic_handle(origin=origin, error=0, measurement_interval=interval, jobs=periodic_workload, job_type='temporal', ci_type='LWA', ci_dataframe=ci_df, experiment_name='demo_2')

    return


def demo_periodic_1():
    cstr = 'LWA'
    full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
    ci_df = pd.read_csv(full_carbon_string, parse_dates=True)

    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')

    periodic_workload = temporalWorkload.periodic_temporal_workload(dur=30, ori=origin)
    periodic_workload.loc[:, 'start_timestamp'] -= origin
    periodic_workload = periodic_workload.drop(columns=['location', 'duration'])

    intervals = [30, 60]
    window_sizes = [17, 8]

    for y in range(0,2):
        i = intervals[y]
        win = window_sizes[y]

        lwa_periodic_handle(origin=origin, error=0, measurement_interval=i, jobs=periodic_workload, job_type='temporal', ci_type=cstr, ci_dataframe=ci_df, max_steps_window=win, experiment_name=f'demo_1_i{i}')


def demo_adhoc_1():
    cstr = 'LWA'
    full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
    ci_df = pd.read_csv(full_carbon_string, parse_dates=True)
    ci_df['datetime'] = pd.to_datetime(ci_df['datetime'], format='%Y-%m-%d %H:%M:%S')
    ci_df = ci_df.set_index(['datetime'])

    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    interval = 30

    end_dates = [datetime.strptime("2020-6-30 23:30:00", '%Y-%m-%d %H:%M:%S'), datetime.strptime("2020-11-30 23:30:00", '%Y-%m-%d %H:%M:%S')]

    for x in range(0,2):
        e = end_dates[x]
        temp_workload = temporalWorkload.t_workload(jobs = 3387, measurement_interval=interval, start_dt = origin, end_dt = e)
        temp_workload.loc[:, 'start_timestamp'] -= origin
        y = temp_workload.drop(columns=['location'])

        exp_end = e+timedelta(days=30)
        ci_temp = ci_df.loc[origin:exp_end]

        i = "half" if x==0 else "full"
        lwa_adhoc_handle(origin=origin,measurement_interval=interval,ci_dataframe=ci_temp, ci_type=cstr, ml_jobs=y, job_type='temporal', forecast_method=0, interruptible=False, error=False, experiment_name=i)
    return


def demo_adhoc_2():
    interval = 60
    c_strs = ['LWA', 'marginal', 'watttime']
    origins = ["2020-01-01 00:00:00", "2022-01-01 00:00:00", "2022-01-01 00:00:00"]
    ends = ["2020-01-31 23:00:00", "2022-01-31 23:00:00", "2022-01-31 23:00:00"]
    for x in range(0,3):
        cstr = c_strs[x]
        full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
        if x == 0:
            ci_df = pd.read_csv(full_carbon_string, parse_dates=True)
        else:
            ci_df = pd.read_csv(full_carbon_string, parse_dates=True)[['datetime','FR','US-CAL-CISO','DE','GB']]
        ci_df['datetime'] = pd.to_datetime(ci_df['datetime'], format='%Y-%m-%d %H:%M:%S')
        ci_df = ci_df.set_index(['datetime'])

        o = origins[x]
        e = ends[x]
        origin = datetime.strptime(o, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        
        temp_workload = temporalWorkload.t_workload(jobs = 282, measurement_interval=interval, start_dt = origin, end_dt = end-timedelta(days=5))
        temp_workload.loc[:, 'start_timestamp'] -= origin
        y = temp_workload.drop(columns=['location'])

        ci_temp = ci_df.loc[origin:end]

        lwa_adhoc_handle(origin=origin,measurement_interval=interval,ci_dataframe=ci_temp, ci_type=cstr, ml_jobs=y, job_type='temporal', forecast_method=0, interruptible=False, error=False, experiment_name=f'demo4_{cstr}')
    return


def demo_carbmin_2():
    cstr = 'watttime'
    full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
    ci_df = pd.read_csv(full_carbon_string, parse_dates=True)
    ci_df['datetime'] = pd.to_datetime(ci_df['datetime'], format='%Y-%m-%d %H:%M:%S')
    ci_df = ci_df.set_index(['datetime'])

    origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    interval = 60
    ci_df = interval_selecter(ci_df, interval)
    ci_df = ci_df.reset_index()

    temp_workload = genAI.workloadOverADay(requestsADay=80825, total_hours=743, start_date=origin)
    temp_workload = temp_workload.drop(columns=['location', 'duration'])

    networks = ['AWS', 'Azure', 'GCP', 'base']
    for net in networks:
        if net == 'base':
            ci_temp = ci_df
        else:
            ci_temp = network_handler.network_selecter(carbon_df=ci_df, networks=[net])
        print(net)
        print(ci_temp.columns)
        print(ci_temp)
        
        carbonSum, outputReqs, outputZones = position_dependent_workload(temp_workload, interval, carbon_df=ci_temp)

        carbonMin_output_formatter(outputReqs, outputZones,job_type='gen_AI', ci_type=cstr, experiment_name=net)
    return


def demo_carbmin_1():
    workloads = ['temporal', 'gen_AI']

    cstr = 'marginal'
    full_carbon_string = f'./Data/post_processing/{cstr}_ci.csv'
    ci_df = pd.read_csv(full_carbon_string, parse_dates=True)
    ci_df['datetime'] = pd.to_datetime(ci_df['datetime'], format='%Y-%m-%d %H:%M:%S')
    ci_df = ci_df.set_index(['datetime'])
    origin = datetime.strptime("2022-06-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime("2022-6-30 23:00:00", '%Y-%m-%d %H:%M:%S')
    ci_df = ci_df.loc[origin:end_date]
    interval = 60
    end_index = len(ci_df.index)

    ai_workload= genAI.workloadOverADay(requestsADay=80825, total_hours=end_index, start_date=origin)
    ai_workload = ai_workload.drop(columns=['location', 'duration'])

    temp_workload = temporalWorkload.t_workload(jobs = 2424750, start_dt = origin, end_dt = end_date, loc="FR", min_duration=60, max_duration=120, measurement_interval = interval)
    temp_workload = temp_workload.drop(columns=['location', 'duration'])

    for w in workloads:
        if w == 'gen_AI':
            workload = ai_workload
        else:
            workload = temp_workload

        carbonSum, outputReqs, outputZones = position_dependent_workload(workload, interval, carbon_df=ci_df.reset_index())
        carbonMin_output_formatter(outputReqs, outputZones,job_type=w, ci_type=cstr, experiment_name='demo5')

    return


demo_periodic_2()
#carbonDF = network_handler.network_selecter(carbon_df=carbonDF, networks=['AWS', 'Azure'])
#print(carbonDF.columns)

#countryDF = pd.read_csv('./Workloads/carbonMin_paper_country_data.csv')
#mini_carbondf = carbonDF[['datetime', 'IN-WE','NL','US-TEX-ERCO','FR','US-CAL-CISO','JP-TK','DE','GB']]
#carbonSum = genAI_workload(interval=60, start_day='2023-01-01', end_index=len(mini_carbondf.index), countryDF=countryDF, carbon_df=mini_carbondf)

#carbonSum = temporal_workload(30, 3387, 576, 21120)
#workload = temporalWorkload.t_workload(jobs = 3387, start_dt = 0, end_dt = 744, loc="FR", measurement_interval=30)
#print(carbonSum)

#temporalWorkload.temporal_workload(80825)

#z_date = datetime.strptime('2022-01-01', '%Y-%m-%d').date()
#x = genAI.workloadOverADay(requestsADay=8005, start_date=z_date, end_index=1400)
#x = temporalWorkload.t_workload(jobs = 80825, start_dt = 0, end_dt = 1400, loc="FR")
#x = Azure_processing.AzureFunctionsInvocationTraceForTwoWeeksJan2021()
#x = Azure_processing.AzureFunctions2019_total_workload()

#y = carbon_min_algorithm.carbonMin_API_conversion(x)

#temp_workload = temporalWorkload.t_workload(jobs = 3387, start_dt = 576, end_dt = 21120, measurement_interval=30)
"""
origin = datetime.strptime("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')  #temp_workload['start_timestamp'].iloc[0]

#temp_workload = temporalWorkload.t_workload(jobs = 3387, measurement_interval=30, start_dt = datetime.strptime("2020-01-01", '%Y-%m-%d'), end_dt = datetime.strptime("2020-12-31", '%Y-%m-%d'))
temp_workload = temporalWorkload.periodic_temporal_workload(dur=30, ori=origin)


temp_workload.loc[:, 'start_timestamp'] -= origin

#y = temp_workload.drop(columns=['location'])
y = temp_workload.drop(columns=['location', 'duration'])

carbonDF['datetime'] = pd.to_datetime(carbonDF['datetime'], format='%Y-%m-%d %H:%M:%S')
ci_df = carbonDF.set_index(['datetime'])

ci_df = ci_df[origin:]
#ci_df = ci_df["2020-01-01 00:00:00":]#datetime.strptime("2020-01-01", '%Y-%m-%d'):]
"""
"""
interr = [True, False]
errors = [0, 0.05, 0.1]
forcastmethods = ["next_workday", "semi_weekly"]
for i in interr:
    for error in errors:
        for forecast_method in forcastmethods:
            LWA_algorithm.ml_experiment(measurement_interval=30,ci_dataframe=ci_df, ci_type=carbon_string, ml_jobs=y, job_type='temporal', forecast_method=forecast_method, interruptible=i, error=error)
LWA_algorithm.ml_experiment(measurement_interval=30,ci_dataframe=ci_df, ci_type=carbon_string, ml_jobs=y, job_type='temporal', forecast_method=0, interruptible=False, error=False)
"""
#errors = [0.05] #0]
#for error in errors:
#     LWA_algorithm.periodic_experiment(error=error, measurement_interval=30, jobs=y, ci_dataframe=ci_df)

print('finished')

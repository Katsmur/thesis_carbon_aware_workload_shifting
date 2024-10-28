import pandas as pd
import numpy as np
from datetime import timedelta
import os
import sys
from typing import List, Tuple

currdir = os.path.dirname(__file__)
workload_module = os.path.join(currdir,"../../",  "Workloads")
sys.path.append(workload_module)
import workload_API

class LetsWaitAwhile_job(workload_API.Job):
    def __init__(self, loc, time, dur):
        self.loc = loc
        self.time = time
        self.dur = dur
    
    def origin_location(self):
        return self.loc
    
    def start_time(self):
        return self.time

    def duration(self):
        return self.dur


def t_workload(jobs, 
               start_dt = "2020-01-01", 
               end_dt = "2020-12-31", 
               loc=False, 
               workday_start = 9*60, 
               workday_end = 17*60, 
               min_duration=4*60, 
               max_duration=90*60, 
               measurement_interval = 60, 
               workdays_only = True):

    if not loc:
        loc = 'N/A'
    rng = np.random.default_rng(0)
    possible_durations = range(min_duration, max_duration, measurement_interval)
    possible_start_times = range(workday_start, workday_end, measurement_interval)

    days = pd.date_range(start_dt, end_dt, freq="D")
    if workdays_only:
        usable_days = [day for day in days if 0 <= day.weekday() <= 4]
    else:
        usable_days = days
    jobs_per_day = rng.multinomial(jobs, [1 / len(usable_days)] * len(usable_days))
    
    s = pd.Series(jobs_per_day, index=usable_days).reindex(days, fill_value=0)
    start_dates = []
    for index, value in s.items():
        ds = [index] * value
        start_minutes = np.sort(rng.choice(possible_start_times, size=len(ds)))
        for d, start_minute in zip(ds, start_minutes):#, durations):
            dt = d + timedelta(minutes=int(start_minute))
            start_dates.append(dt)
    durations = rng.choice(possible_durations, size=jobs)

    list_of_jobs = []
    for start_date, dur in zip(start_dates, durations):
        duration = timedelta(minutes=int(dur))
        list_of_jobs.append([start_date, duration])

    df = pd.DataFrame(list_of_jobs, columns=['start_timestamp', 'duration'])
    df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'num_of_requests'})
    df['location'] = pd.Series([loc for x in range(len(df.index))])
    api_object = workload_API.Job_DF(df)
    return api_object

def periodic_temporal_workload(ori, dur, loc=False, days=365, start_minute_each_day=60):
    if not loc:
        loc = 'N/A'
    # daily at 1am
    jobs = [((time + start_minute_each_day)*60, dur, loc, 1) for time in range(0, 1440 * days, 1440)]
    temp = pd.DataFrame(jobs, columns=['start_timestamp', 'duration', 'location', 'num_of_requests'])
    temp.start_timestamp = pd.to_datetime(temp.start_timestamp,  origin=ori, unit='s')
    temp.duration = pd.to_timedelta(temp.duration, unit='m')
    api_object = workload_API.Job_DF(temp, columns=['start_timestamp', 'duration', 'location', 'num_of_requests'])
    return api_object


def temporal_workload(daily_workload):
    w_list = t_workload(daily_workload)
    df = pd.DataFrame(w_list, columns=['datetime', 'jobs'])
    save_name = os.path.join('./', 't_workload.csv')
    df.to_csv(save_name, index=False)
    print('Saved Temporal Workload')

#temporal_workload()
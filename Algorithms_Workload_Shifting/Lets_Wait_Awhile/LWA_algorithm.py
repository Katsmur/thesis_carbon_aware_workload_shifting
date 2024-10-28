from multiprocessing import Pool
from typing import List, Iterator, Tuple

import numpy as np
import pandas as pd
import simpy
import os
from datetime import timedelta

from leaf.infrastructure import Infrastructure, Node
from leaf.power import PowerMeter, PowerModelNode

from strategy import Strategy, BidirectionalStrategy, AdHocStrategy

#MEASUREMENT_INTERVAL = 60  # mins
ERROR_REPETITIONS = 10  # Repetitions for each run with random noise that simulates forecast errors

Job = Tuple[int, int]  # arrival time, duration


def main(node, ci, jobs: List[Job], strategy: Strategy, measurement_interval):
    env = simpy.Environment()
    env.process(datacenter_process(env, jobs, strategy))
    power_meter = PowerMeter(node, measurement_interval=measurement_interval)
    env.process(power_meter.run(env, delay=0.01))
    # print(f"Starting simulation with strategy {strategy}...")
    env.run(until=len(ci) * measurement_interval)

    # Watt usage at point in time
    return np.array([float(measurement) for measurement in power_meter.measurements])


def _print_analysis(consumption_timeline, ci, measurement_interval):
    consumed_kwh = sum(consumption_timeline) * measurement_interval / 60 / 1000

    # gCO2/timestep = gCO2/kWh * W / 1000 / steps_per_hour
    gco2_per_timestep = np.multiply(ci.values, consumption_timeline) * measurement_interval / 60 / 1000
    emitted_gco2 = sum(gco2_per_timestep)

    print(f"- Consumed {consumed_kwh:.2f} kWh and emitted {emitted_gco2 / 1000:.2f} kg CO2e.")
    print(f"- Average CO2 intensity of used energy was {emitted_gco2 / consumed_kwh:.2f} gCO2e/kWh.")


def datacenter_process(env: simpy.Environment, jobs: Iterator[Job], strategy: Strategy):
    for arrival_time, duration in jobs:
        time_until_arrival = arrival_time - env.now
        if isinstance(strategy, BidirectionalStrategy):
            time_until_arrival -= strategy.window_in_minutes
        if time_until_arrival < 0:
            continue
        yield env.timeout(time_until_arrival)
        env.process(strategy.run(env, duration))


def adhoc_worker(node, jobs, ci, error, seed, forecast_method, interruptible, measurement_interval):
    strategy = AdHocStrategy(node=node, ci=_apply_error(ci, error, seed),
                             interval=measurement_interval, forecast=forecast_method, interruptible=interruptible)
    return main(node, ci, jobs, strategy, measurement_interval)


def bidirectional_worker(node, jobs, ci, error, seed, window, measurement_interval, country):
    strategy = BidirectionalStrategy(node=node, ci=_apply_error(ci, error, seed),
                                     window_in_minutes=window * measurement_interval, interval=measurement_interval)
    consumption_timeline = main(node, ci, jobs, strategy, measurement_interval)
    
    #x = pd.DataFrame()
    #x['time'] = ci.index
    #x['jobs'] = consumption_timeline
    #x.to_csv(f'./recreateExp/{country}/periodic_{error}_{seed}_{window}_jobs.csv', index=False)

    return ci[consumption_timeline == 1].sum() / 365


def periodic_experiment(error, measurement_interval, jobs, ci_dataframe, max_steps_window: int = 17, experiment_name=False):
    print(f"Each Job is assumed to have a duration equal to the measurement interval of {measurement_interval} minutes.")
    LWA_periodic_workload_verifier(jobs, measurement_interval)
    jobs = LWA_API_conversion(jobs, measurement_interval, adhoc=False)
    result = {}
    
    countries = ci_dataframe.columns
    for country in countries:
        if experiment_name:
            print(f"Running {experiment_name} experiment for {country} with error {error}.")
        else:
            print(f"Running periodic experiment for {country} with error {error}.")
        ci = ci_dataframe[country]
        ci.dropna(inplace=True)
        ci = ci[ci.index.minute % measurement_interval == 0]
            
        infrastructure = Infrastructure()
        node = Node("dc", power_model=PowerModelNode(power_per_cu=1))
        infrastructure.add_node(node)

        window_results = []
        for window in range(max_steps_window):
            if error:
                bidirectional_args = ((node, jobs, ci, error, seed, window, measurement_interval, country) for seed in range(ERROR_REPETITIONS))
                with Pool(ERROR_REPETITIONS) as pool:
                    repeat_results = pool.starmap(bidirectional_worker, bidirectional_args)
                mean_result = np.array(repeat_results).mean()
                window_results.append(mean_result)
            else:
                window_results.append(bidirectional_worker(node, jobs, ci, error, None, window, measurement_interval, country))
        result[country] = window_results

    return result


def ml_experiment(measurement_interval, ci_dataframe, ml_jobs, forecast_method, interruptible: bool=False, error: float=False, specific_region: str=False):
    LWA_adhoc_workload_verifier(ml_jobs, measurement_interval)
    ml_jobs = LWA_API_conversion(ml_jobs, measurement_interval, adhoc=True)
    #ci_dataframe['datetime'] = pd.to_datetime(ci_dataframe['datetime'], format='%Y-%m-%d %H:%M:%S')
    #ci_dataframe.set_index(['datetime'], inplace=True)
    #ci_dataframe = ci_dataframe.asfreq(freq=timedelta(minutes=int(measurement_interval)), method='ffill')
    if specific_region:
        countries = [specific_region]
    else:
        countries = ci_dataframe.columns
        
    for country in countries:
        ci = ci_dataframe[country]
        ci = ci.dropna()
        ci = ci[ci.index.minute % measurement_interval == 0]

        return ad_hoc_experiment(experiment_name="ml", 
                            measurement_interval=measurement_interval, 
                            jobs=ml_jobs, 
                            ci=ci,
                            forecast_method=forecast_method,
                            interruptible=interruptible,
                            error=error)


def ad_hoc_experiment(experiment_name: str,
                      measurement_interval: int,
                      jobs: List,
                      ci,
                      forecast_method,
                      interruptible: bool,
                      error: float):
    print(f"ad_hoc_experiment({experiment_name}, forecast_method={forecast_method}, "
          f"interruptible={interruptible}, error={error})")

    # Build infrastructure
    infrastructure = Infrastructure()
    node = Node("dc", power_model=PowerModelNode(power_per_cu=1))
    infrastructure.add_node(node)

    # Run experiment(s)
    if error:
        adhoc_args = ((node, jobs, ci, error, seed, forecast_method, interruptible, measurement_interval) for seed in range(ERROR_REPETITIONS))
        with Pool(ERROR_REPETITIONS) as pool:  # from 0 to +-8h in 30min intervals (16 experiments)
            worker_results = pool.starmap(adhoc_worker, adhoc_args)
            timeline = np.mean(worker_results, axis=0)
    else:
        timeline = adhoc_worker(node, jobs, ci, error, None, forecast_method, interruptible, measurement_interval)

    # _print_analysis(timeline, ci)

    return timeline, ci


def LWA_adhoc_workload_verifier(workload, measurement_interval):
    assert isinstance(workload, pd.DataFrame)
    expected_columns = set(['start_timestamp', 'duration', 'num_of_requests'])
    actual_columns = set(workload.columns)
    missing_cols = expected_columns - actual_columns
    extra_cols = actual_columns - expected_columns
    if extra_cols:
        raise AttributeError(f"This Algorithm only uses {expected_columns} columns in the Workload")
    if missing_cols:
        raise AttributeError(f"Columns {missing_cols} are missing in the DataFrame")
    
    #Checks if the workload was set to start from the origin of the carbon intensity data so it starts from year 0 and is in the correct units
    assert workload.start_timestamp.dtype =='timedelta64[ns]'
    #Checks if the workload timestamp is in minutes
    assert all(workload['start_timestamp'].dt.components.seconds == 0)
    #Checks if the workload duration is in the correct units
    assert workload.duration.dtype =='timedelta64[ns]'
    #Checks if the workload duration is in minutes
    assert all(workload['duration'].dt.components.seconds == 0)
    #Checks if the workload timestamp interval matches the measurement_interval
    #assert workload['start_timestamp'].iloc[1].components.minutes == measurement_interval
    #work_list = workload.start_timestamp.tolist()
    #for i in range(len(work_list)-1):
    #    difference = abs(work_list[i+1].components.minutes - work_list[i].components.minutes)
    #    if(difference != 0 and difference != measurement_interval):
    #        raise ValueError(f'The timestamp interval of {difference} minutes in the workload does not match the algorithm interval of {measurement_interval} minutes')
    assert all(workload['start_timestamp'].dt.components.minutes % measurement_interval == 0)
    assert all(workload['duration'].dt.components.minutes % measurement_interval == 0)
    

def LWA_periodic_workload_verifier(workload, measurement_interval):
    assert isinstance(workload, pd.DataFrame)
    expected_columns = set(['start_timestamp', 'num_of_requests'])
    actual_columns = set(workload.columns)
    missing_cols = expected_columns - actual_columns
    extra_cols = actual_columns - expected_columns
    if extra_cols:
        raise AttributeError(f"This Algorithm only uses {expected_columns} columns in the Workload")
    if missing_cols:
        raise AttributeError(f"Columns {missing_cols} are missing in the DataFrame")
    
    #Checks if the workload was set to start from the origin of the carbon intensity data so it starts from year 0 and is in the correct units
    assert workload.start_timestamp.dtype =='timedelta64[ns]'
    #Checks if the workload timestamp is in minutes
    assert all(workload['start_timestamp'].dt.components.seconds == 0)
    #Checks if the workload timestamp interval matches the measurement_interval
    assert all(workload['start_timestamp'].dt.components.minutes % measurement_interval == 0)


def LWA_API_conversion(original_workload, measurement_interval, adhoc=True):
    workload = original_workload.copy()
    if adhoc:
        workload.duration = (workload.duration.dt.total_seconds()//60).astype(int)
    workload.start_timestamp = (workload.start_timestamp.dt.total_seconds()//60).astype(int)
    """
    workload = workload.drop(columns=['location'])
    workload['start_timestamp'] = workload['start_timestamp'].view(int)//1e9
    workload.loc[:, 'start_timestamp'] //= 60
    origin = workload['start_timestamp'].iloc[0]
    workload.loc[:, 'start_timestamp'] -= origin
    workload['start_timestamp'] = workload['start_timestamp'].astype(int)"""

    jobs = []
    if adhoc:
        for row in workload.itertuples(index=False):
            jobs.extend([[row.start_timestamp, row.duration]]*row.num_of_requests)
    else:
        for row in workload.itertuples(index=False):
            jobs.extend([[row.start_timestamp, measurement_interval]]*row.num_of_requests)
    return jobs


def _apply_error(ci, error, seed):
    if error is None:
        return ci
    rng = np.random.default_rng(seed)
    return ci + rng.normal(0, error * ci.mean(), size=len(ci))


if __name__ == '__main__':
    # Scenario I
    print("Starting Scenario 1...")
    #periodic_experiment(error=0)
    #periodic_experiment(error=0.05)

    # Scenario II
    print("Starting Scenario 2...")
    ml_experiment()
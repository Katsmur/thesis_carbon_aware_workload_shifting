import pandas as pd
from datetime import timedelta
#computation_per_request = 0     #imported from opEmissions.py

#requests_shifted = 0            #number of requests shifted to given region
#resource_capacity = 0           #GPU-seconds?   resource capacity of given datacenter
#headroom_capacity = 0           #headroom capacity of datacenter today, probably irrelevant for this project

#datacenter_utilization = requests_shifted * opEmissions.computation_per_request()
#datacenter_cap = resource_capacity + headroom_capacity


#Computation (GPU-seconds) per request
def computation_per_request():
    #? copied straight from paper
    gpu_capacity = 156                  #TFLOPS is the GPU capacity assuming 50% efficiency
    word_count = 185                    #the output word count (measured average of 185 output words/request)
    inference_per_word = 5              #number of inferences per output word (assumed window/sampling of 5 for each output word)
    operation_inference = 0.35          #TFLOPS per inference assuming GPT-3 model (around 175 billion weights) processed with BF16 operations

    return ((operation_inference * inference_per_word * word_count) / gpu_capacity)

#Energy consumed for serving inference requests
def operational_carbon_emissions(carbon_metric, requests_in_region):
    power_utilization_efficiency = 1.1  #average of google data centers, need to change depending on specific cloud platform
    thermal_design_power = 0.428        #kW per GPU (1/8 of 3.43 kW for the instance)   not certain if extended acronym is correct
    
    energy_used_by_inferences = requests_in_region * computation_per_request() * thermal_design_power * power_utilization_efficiency
    return carbon_metric * energy_used_by_inferences


def embodied_carbon_emissions(requests_in_region, resource_capacity):
    hardware_lifespan = 94608000        #seconds in a 3 year lifespan
    per_gpu_emissions = 318/1000        #grams of CO2 emissions per GPU (1/8 of per instance emissions)
    
    service_time = resource_capacity * requests_in_region * computation_per_request()
    return (service_time / hardware_lifespan) * per_gpu_emissions

#Greedy Algorithm CarbonMin
#choose datacenter with the lowest carbon metric whose datacenter_utilization + current workload <= datacenter_cap

def carb_min_method(carbon_metrics, reqs_at_hour):
    datacenter_regions = len(carbon_metrics)
    #datacenter_utilization = reqs_at_hour * opEmissions.computation_per_request()
    resource_capacity = (reqs_at_hour/datacenter_regions)/0.3

    reqs_per_used_zones = []
    i = 0
    op_carbon_sum = 0
    em_carbon_sum = 0
    while(reqs_at_hour > 0):
        reqs_per_used_zones.append(carbon_metrics.index.values[i])
        if(reqs_at_hour - resource_capacity > 0):
            op_carbon_sum += operational_carbon_emissions(carbon_metrics.iloc[i], resource_capacity) 
            em_carbon_sum += embodied_carbon_emissions(resource_capacity, resource_capacity)
            reqs_per_used_zones.append(resource_capacity)
        else:
            op_carbon_sum += operational_carbon_emissions(carbon_metrics.iloc[i], reqs_at_hour) 
            em_carbon_sum += embodied_carbon_emissions(reqs_at_hour, resource_capacity)
            reqs_per_used_zones.append(reqs_at_hour)
        
        reqs_at_hour -= resource_capacity
        i += 1
        
    op_em_reqsPerZone = [op_carbon_sum, em_carbon_sum] + reqs_per_used_zones
    #reqs_per_used_zones.insert(0, em_carbon_sum)
    #reqs_per_used_zones.insert(0, op_carbon_sum)
    return op_em_reqsPerZone


def carbonMin_verifier(workload, measurement_interval):
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
    assert workload.start_timestamp.dtype =='datetime64[ns]'
    #Checks if the workload timestamp is at least in minutes
    assert all(workload['start_timestamp'].dt.second == 0)
    #Checks if the workload timestamp interval matches the measurement_interval
    assert all(workload['start_timestamp'].dt.minute % measurement_interval == 0)


def carbonMin_API_conversion(workload, interval):
    #workload.start_timestamp = pd.to_datetime(workload.start_timestamp)
    reduced_work = workload.set_index('start_timestamp').resample(timedelta(minutes=int(interval))).num_of_requests.sum().reset_index()
    #reduced_work.start_timestamp = reduced_work.start_timestamp.astype(str)
    output = reduced_work.values.tolist()
    """
    workload = workload.to_df().values.tolist()
    output = [["N/A", 0]]
    for i in workload:
        if output[-1][0][0:13] in i[0]:
            output[-1][1] += i[3]
        else:
            output.append([i[0][0:13]+':00:00', i[3]])
    output.pop(0)
    """
    return output
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

currdir = os.path.dirname(__file__)
country_module = os.path.join(currdir,"../../",  "Workloads")
sys.path.append(country_module)
import country_users
import workload_API


def workloadOverADay(requestsADay, start_date=datetime.strptime('2019-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), total_hours=765, countryDF=pd.DataFrame()):

    diurnalStructure = [0.009, 0.005, 0.004, 0.003, 0.003, 0.005, .021, .039, .05, .055, .06, .061, .062, .061, .061, .062, .062, .063, .063, .063, .063, .059, .044, .022]

    if countryDF.empty:
        countryDF = country_users.country_processing()

    def countryWorkloadAtHour(hour):

        gpu_capacity = 156                  #TFLOPS is the GPU capacity assuming 50% efficiency
        word_count = 185                    #the output word count (measured average of 185 output words/request)
        inference_per_word = 5              #number of inferences per output word (assumed window/sampling of 5 for each output word)
        operation_inference = 0.35          #TFLOPS per inference assuming GPT-3 model (around 175 billion weights) processed with BF16 operations
        gpu_secs_per_req = timedelta(seconds=((operation_inference * inference_per_word * word_count) / gpu_capacity))

        day_hour = hour % 24
        datetime_ind = start_date + timedelta(hours=hour)
        list_of_jobs = []
        for row in countryDF.values.tolist():
            
            lochour = country_users.hour_localization(day_hour, row[2])
                
            localActivity = diurnalStructure[lochour]

            countryReqHour = int(round(localActivity * requestsADay * row[3], 0))
            
            list_of_jobs.append([datetime_ind, gpu_secs_per_req, row[1], countryReqHour])
            
        return list_of_jobs

    reqsAtHour = []
    for hour in range(total_hours):
        reqsAtHour += countryWorkloadAtHour(hour)
    
    api_object = workload_API.Job_DF(reqsAtHour, columns=['start_timestamp', 'duration', 'location', 'num_of_requests'])
    
    return api_object
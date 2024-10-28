import pandas as pd
import os
import country_converter as coco
from datetime import datetime
import pytz
import math

def hour_localization(utc, add_hour):
    lochour = utc + add_hour
    if lochour < 0:
        lochour += 24
    elif lochour > 23.6:
        lochour -= 24
    return math.floor(lochour)

def country_processing():
    cc = coco.CountryConverter()
    
    countriesfile = './Workloads/CountryUsersCSV.csv'
    #countriesfile = './CountryUsersCSV.csv'
    countryUsersDF = pd.read_csv(countriesfile)

    #countryUsersDF.drop(countryUsersDF.tail(1).index,inplace=True)

    countryUsersDF.columns.values[0] = 'countries' #= countryUsersDF.rename(columns={'Most Active Countries on Youtube and Facebook': 'countries', 'Social Media Users in 2023 (millions)': 'users'})
    countryUsersDF.columns.values[1] = 'users'
    sumUsers = round(countryUsersDF['users'].sum(), 2)

    countryUsersDF.insert(1, "code", cc.pandas_convert(series=countryUsersDF['countries'], to='ISO2'), True)

    timezones = []
    for country in countryUsersDF['code']:
        tzs = pytz.country_timezones(country)
        tz = pytz.timezone(tzs[0])
        if country == 'Brazil':
            tz = pytz.timezone(tzs[7])
        aware = datetime.now(tz)
        timezones.append(aware.utcoffset().total_seconds() / (60*60))

    countryUsersDF.insert(2, "UTC", timezones, True)

    countryPortion = []
    for users in countryUsersDF['users']:
        countryPortion.append(users/sumUsers)

    countryUsersDF.insert(3, "percent", countryPortion, True)

    return countryUsersDF
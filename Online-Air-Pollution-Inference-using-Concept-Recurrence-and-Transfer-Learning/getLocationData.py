import numpy as np
import pyreadr
import pandas as pd
import matplotlib
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime
from datetime import timedelta

def roundTime(dt=None, roundTo=60):
   if dt == None : dt = datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)


def filter_location(df_all, df_sensor, place):
    place_sensors = df_sensor.loc[df_sensor['location'] == place]
    place_sensors = list(place_sensors.serialn)
    #place_sensors = list(set(place_sensors))

    place_df = df_all[df_all['serialn'].isin(place_sensors)]
    place_df.drop(["GAS1", "Tgas1", "GAS2", "timestamp", "tags"], axis=1, inplace=True)
    place_df.dropna(inplace=True)
    place_df['date']=place_df['date'].dt.round('min')  
    place_df.sort_values(by="date", inplace=True)
    name = place.split(" ")[0]
    #print(name)
    if name == "":
        name = place.split(" ")[1]
    # print(name, place_df.shape)
    # boolean = not place_df["date"].is_unique
    # print("Has dups?", boolean)
    # place_df.drop_duplicates(subset ="date",
    #                  keep = False, inplace = True)
    # boolean = not place_df["date"].is_unique
    print(name, len(set(place_sensors)))
    print(list(place_df.date))
    print(list(place_df.date)[0], list(place_df.date)[-1])
    print(name, place_df.shape)
    place_df.to_csv(name + " data.csv", index=False)




result = pyreadr.read_r('odin.RData')
df_all = result['odin_data']

df_sensor = pd.read_csv('sensor location.csv')
#places = list(set(list(df_sensor.location)))

#########   Arrow Town   #########
place = "Arrowtown"
filter_location(df_all, df_sensor, place)

#########   Invercargill   #########
place = " Invercargill City Southland New Zealand / Aotearoa"
# filter_location(df_all, df_sensor, place)

#########   Masterton   #########
place = "Masterton"
filter_location(df_all, df_sensor, place)

#########   Cromwell   #########
place = "Cromwell Community"
filter_location(df_all, df_sensor, place)

#########   Reefton   #########
place = "Reefton"
filter_location(df_all, df_sensor, place)




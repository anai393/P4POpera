# Install using pip
import pyreadr
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
from datetime import timedelta


def get_filter_locations(df):
    coord = df[["lat", "lon", "serialn"]].dropna()
    coord = coord.drop_duplicates()
    # print(coord.shape)
    coord["coord"] = coord[["lat", "lon"]].apply(tuple, axis=1)

    # get the geological locations based on the coordinates using the Nominatim geocoding services
    geolocator = Nominatim(user_agent="Something")
    # Since we are uinsg the free Nominatim geocoding services, we need to limit how fast we send a query otherwise we get kicked.
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    place = []
    
    for c in coord["coord"].tolist():
        Latitude = str(c[0])
        Longitude = str(c[1])
        location = geocode(Latitude+","+Longitude, addressdetails=True)
        data = location.raw
        # print(data)
        data = data['address']
        # print(data)
        town = ''
        address = str(data)
        if 'county' not in data:
            print(data, 'None')
            place.append('None')
            continue
        county = str(data['county'])

        if 'town' in data:
            town = str(data['town'])
        state = str(data['state'])
        country = str(data['country'])
        address = town + ' ' + county + ' ' + state + ' ' + country
        
        if(address != 'None'):
            place.append(address)
        print(address)

    coord["location"] = place
    coord.to_csv("sensor location.csv", index=False)


def get_invercargill_locations(df):
    coord = df[["lat", "lon", "serialn"]].dropna()
    coord = coord.drop_duplicates()
    # print(coord.shape)
    coord["coord"] = coord[["lat", "lon"]].apply(tuple, axis=1)

    # get the geological locations based on the coordinates using the Nominatim geocoding services
    geolocator = Nominatim(user_agent="Something")
    # Since we are uinsg the free Nominatim geocoding services, we need to limit how fast we send a query otherwise we get kicked.
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    place = []
    
    for c in coord["coord"].tolist():
        Latitude = str(c[0])
        Longitude = str(c[1])
        location = geocode(Latitude+","+Longitude, addressdetails=True)
        data = location.raw
        # print(data)
        data = data['address']
        # print(data)
        town = ''
        address = str(data)
        if 'county' not in data:
            print(data, 'None')
            city = str(data['city'])
            state = str(data['state'])
            country = str(data['country'])
            address = city + ' ' + state + ' ' + country
            place.append(address)
            continue
        county = str(data['county'])

        if 'town' in data:
            town = str(data['town'])
        state = str(data['state'])
        country = str(data['country'])
        address = town + ' ' + county + ' ' + state + ' ' + country
        
        if(address != 'None'):
            place.append(address)
        print(address)

    coord["location"] = place
    coord.to_csv("sensor location.csv", index=False)


def filter_location(df, df_sensor, place):
    # Get data with sensors located in the given place
    place_sensors = df_sensor.loc[df_sensor['location'] == place]
    place_sensors = list(place_sensors.serialn)
    #place_sensors = list(set(place_sensors))
    place_df = df[df['serialn'].isin(place_sensors)]

    # Remove useless features
    place_df.drop(["GAS1", "Tgas1", "GAS2", "timestamp",
                  "tags"], axis=1, inplace=True)
    place_df.dropna(inplace=True)
    place_df['date'] = place_df['date'].dt.round('min')
    place_df.sort_values(by="date", inplace=True)
    name = place.split(" ")[0]

    if name == "":
        name = place.split(" ")[1]

    print(name, len(set(place_sensors)))
    print(list(place_df.date)[0], list(place_df.date)[-1])
    print(name, place_df.shape)
    place_df.to_csv(name + " data.csv", index=False)


"""
Original data description:
It's a 1 minute time resolution R dataset with one data.frame in it:
odin_data. The PM data in this dataset has NOT been corrected or calibrated.
The features are:
"PM1": Particulate matter of size smaller than 1 micron
"PM2.5": Particulate matter of size smaller than 2.5 micron
"PM10": Particulate matter of size smaller than 10 micron
"PMc": Particulate matter with size between 2.5 and 10 microns
"GAS1": Placeholder -- no valid data
"Tgas1": Placeholder -- no valid data
"GAS2": Placeholder -- no valid data
"Temperature": Temperature inside the ODIN device
"RH": RH inside the ODIN device
"date": Corrected and usable POSIXct date in UTC timezone
"timestamp": RAW timestamp -- POSIXct in UTC timezone
"deviceid": Unique identifier for the devices -- Tied to our data engine backend
"tags": List of tags that group our devices. It includes information about campaigns. There is a dictionary to convert "_TAG_?????_" into "human readable" tag
"lat": Latitude. For that time period. If it is NA, then the location information is not 100% reliable.
"lon": Longitude. For that time period. If it is NA, then the location information is not 100% reliable. 
"serialn": "human readable" serial number for each ODIN device (this is what we use in public facing plots)
"""

# Reads the R data into a pandas dataframe
result = pyreadr.read_r('odin.RData')
df_all = result['odin_data']

# Get the geological locations based on the coordinates of the sensors (data saved as .csv)
# get_filter_locations(df_all)

df_sensor = pd.read_csv('sensor location.csv')

# Locations:
#places = list(set(list(df_sensor.location)))

# Get the locations specific data using the locations of the sensors (New data saved as .csv)
#########   Arrow Town   #########
place = "Arrowtown Queenstown-Lakes District Otago New Zealand/Aotearoa"
# filter_location(df_all, df_sensor, place)

#########   Invercargill   #########
place = "Invercargill City Southland New Zealand/Aotearoa"
# get_invercargill_locations(df_all)
filter_location(df_all, df_sensor, place)

#########   Masterton   #########
place = "Masterton Masterton District Wellington New Zealand/Aotearoa"
# filter_location(df_all, df_sensor, place)

#########   Cromwell   #########
place = "Cromwell Community Central Otago District Otago New Zealand/Aotearoa"
# filter_location(df_all, df_sensor, place)

#########   Reefton   #########
place = "Reefton Buller District West Coast New Zealand/Aotearoa"
# filter_location(df_all, df_sensor, place)


'''
Notes:
1. Data contains a looooooot of missing data, therefore the Timestamps do not always have a uniform gap.
2. Not all sensors are active at all times, many sensors only become active for a limited time.
3. The location of the sensors can be plooted out using plotly, I suggest taking a look at: https://plotly.com/python/maps/
Good luck!
'''

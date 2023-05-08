import math

import pandas as pd


path_meta = 'D:/study/_data/AIFac_pollution/META/'

# read in the csv files

pmmap_csv = pd.read_csv(path_meta+'pmmap.csv', index_col=False, encoding='utf-8')

awsmap_csv = pd.read_csv(path_meta+'awsmap.csv', index_col=False, encoding='utf-8')


# create a dictionary of places in awsmap.csv
places_awsmap = {}

for i, row in awsmap_csv.iterrows():

    places_awsmap[row['Description']] = (row['Latitude'], row['Longitude'])


# define a function to calculate the distance between two points

def distance(lat1, lon1, lat2, lon2):

    R = 6371 # earth radius in kilometers

    dLat = math.radians(lat2 - lat1)

    dLon = math.radians(lon2 - lon1)

    lat1 = math.radians(lat1)

    lat2 = math.radians(lat2)

    a = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


# loop over the locations in pmmap.csv

for i, row_a in pmmap_csv.iterrows():

# find the closest places in awsmap.csv to the location in pmmap.csv

    closest_places = []

    closest_distances = []

    for j, row_b in awsmap_csv.iterrows():

        dist = distance(row_a['Latitude'], row_a['Longitude'], row_b['Latitude'], row_b['Longitude'])

        if len(closest_places) < 3:

            closest_places.append(row_b['Location'])

            closest_distances.append(dist)

        else:

            max_index = closest_distances.index(max(closest_distances))

            if dist < closest_distances[max_index]:

                closest_places[max_index] = row_b['Location']

                closest_distances[max_index] = dist

# print the closest places for the location in pmmap.csv

    print("Closest places to {}: {}, {}, {}".format(row_a['Location'], closest_places[0], closest_places[1], closest_places[2]))


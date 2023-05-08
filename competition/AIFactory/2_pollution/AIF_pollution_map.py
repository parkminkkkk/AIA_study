import folium
import pandas as pd

path_meta = 'D:/study/_data/AIFac_pollution/META/'
awsmap_csv = pd.read_csv(path_meta+'awsmap.csv', index_col=False, encoding='utf-8')
pmmap_csv = pd.read_csv(path_meta+'pmmap.csv', index_col=False, encoding='utf-8')

# 위치 정보
a_latitude = 36.5122519
a_longitude = 127.2467891

# 지도 객체 생성
m = folium.Map(location=[a_latitude, a_longitude], zoom_start=15)

for index, row in pmmap_csv.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    folium.CircleMarker([lat, lon], radius=5, color='red', fill_color='red',popup=row['Location']).add_to(m)

for index, row in awsmap_csv.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    folium.CircleMarker([lat, lon], radius=5, color='blue', fill_color='blue',popup=row['Location']).add_to(m)

# 위치 정보에 해당하는 마커 생성
folium.Marker([a_latitude, a_longitude]).add_to(m)

# 지도 저장
m.save('./_save/map.html')
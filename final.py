# 필요 라이브러리 
import streamlit as st
import folium
from streamlit_folium import folium_static
import json
import pandas as pd 
import numpy as np
import requests
from tensorflow.python.keras.models import load_model
from PIL import Image
import io
import cv2

st.set_page_config(
    page_title="지켜줄게..너의 안전..",
    page_icon="❤️",
    layout="wide"
)

@st.cache
def load_data(file_name,en):
    data = pd.read_csv(file_name, encoding=en)
    return data

## 본문 내용
st.markdown("# :car: 위험도로 내비게이션 :car: ")
st.markdown("+ 출발지와 도착지를 입력해주세요. (도로명 주소)")
st.markdown("+ 적색에 가까워 질 수록 사고위험이 높은 도로를 의미합니다! :warning:")

# 사이드바
with st.sidebar:
    st.markdown("## 🔔 위험도로 예측 서비스 🔔")
    st.markdown("#### 💡 해당 서비스는 더욱 안전한 운행경로를 위한 도로 위험도를 예측하여 보여드립니다.")
    st.markdown("#### 💡경로 선택할 시 빠른 도착시간도 중요하지만 우리들의 **안전**도 중요하다!")
    st.markdown("**************")

# sidebar 지정-------------------------------------
df_경로 = ['교통최적+추천(기본값)',
                        '교통최적+무료우선',
                        '교통최적+최소시간',
                        '교통최적+초보',
                        '교통최적+고속도로우선',
                        '최단거리+유/무료',
                        '이륜차도로우선 (일반도로가 없는 경우 자동차 전용도로로 안내 할 수 있습니다.)',
                        '교통최적+어린이보호구역 회피']
with st.sidebar:
    st.markdown("## 💌 여러분의 안전을 지켜드립니다. 💌")
    st.markdown("**************")
    st.header("🔎검색🔎")
    st.markdown("도로명 주소를 입력해주세요!")
    출발지 = st.text_input('출발지', '서울 마포구 마포대로 33')
    도착지 = st.text_input('도착지', '서울 종로구 세종대로 172')
    경로선택 =  st.selectbox("경로선택",df_경로)
    st.markdown("**************")

# num 구하는 로직
def choice(경로선택):
    if 경로선택 == '교통최적+추천(기본값)':
        num = 0
    elif 경로선택 == '교통최적+무료우선':
        num = 1
    elif 경로선택 == '교통최적+최소시간':
        num = 2
    elif 경로선택 == '교통최적+초보':
        num = 3
    elif 경로선택 == '교통최적+고속도로우선':
        num = 4
    elif 경로선택 == '최단거리+유/무료':
        num = 10
    elif 경로선택 == '이륜차도로우선 (일반도로가 없는 경우 자동차 전용도로로 안내 할 수 있습니다.)':
        num = 12
    else:
        num = 19
    return num

# 도로명주소를 좌표값으로 바꿔주는 로직
def get_location(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + address
    headers = {"Authorization": "KakaoAK c88bcccff9c5bef1a68843ff7083841b"}
    api_json = json.loads(str(requests.get(url,headers=headers).text))
    address = api_json['documents'][0]['address']
    crd = {"lat": str(address['y']), "lng": str(address['x'])}
    address_name = address['address_name']

    return crd


# 티맵 api 로직 구간별 좌표불러오기
def tmap_api(start,end,num):
    url = "https://apis.openapi.sk.com/tmap/routes?version=1&callback=function"

    payload = {
        "tollgateFareOption": 16,
        "roadType": 32,
        "directionOption": 1,
        "endX": end['lng'],
        "endY": end['lat'],
        "endRpFlag": "G",
        "reqCoordType": "WGS84GEO",
        "startX": start['lng'],
        "startY": start['lat'],
        "gpsTime": "20230104153000",
        "speed": 100,
        "uncetaintyP": 1,
        "uncetaintyA": 1,
        "uncetaintyAP": 1,
        "carType": 0,
        "startName": "%EC%9D%84%EC%A7%80%EB%A1%9C%20%EC%9E%85%EA%B5%AC%EC%97%AD",
        "endName": "%ED%97%A4%EC%9D%B4%EB%A6%AC",
        "gpsInfoList": "126.939376564495,37.470947057194365,120430,20,50,5,2,12,1_126.939376564495,37.470947057194365,120430,20,50,5,2,12,1",
        "detailPosFlag": "2",
        "resCoordType": "WGS84GEO",
        "sort": "index",
        "searchOption": num
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "appKey": "l7xx543bd2b77e19411d84d2f757bf2396cd"
    }

    response = requests.post(url, json=payload, headers=headers)
    return json.loads(response.text)


def get_image(df_좌표_2):
    img_list = []
    for i in range(len(df_좌표_2)):
        lon = df_좌표_2["경도"][i]
        lat = df_좌표_2["위도"][i]
        # 관측 시야(Field Of View) - 최대 120 기본값 90
        fov = "120"
        # 카메라 상하 방향 설정 - 범위 -90 ~ 90 기본값 0
        pitch = "-40"
        # 방향 - 범위 0 ~ 360 (0 or 360::북, 180: 남)
        heading = "-45"
        google_api_key = "AIzaSyC0yRcVQrxdB1fUrPEtFX51thBkP6PxHDI"
        url = f"https://maps.googleapis.com/maps/api/streetview?size=400x300&location={lat},{lon}&fov={fov}&heading={heading}&pitch={pitch}&key={google_api_key}"
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        bytes_data = response.content
        img = Image.open(io.BytesIO(bytes_data))
        img_list.append(img)
    return img_list

# 하이퍼 파라미터 셋팅
config = {
    'IMG_SIZE':(80),
    'EPOCHS':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':200,
    'SEED':42
}

# 수집한 이미지 변환
def transform(image):
    image = cv2.resize(image, (config["IMG_SIZE"],config["IMG_SIZE"]), interpolation=cv2.INTER_AREA)
    image = image/255
    return image

start = get_location(출발지)
end = get_location(도착지)
num = choice(경로선택)
json_df = tmap_api(start,end,num)


##
beta = []
for i in range(len(json_df['features'])):
    if type(json_df['features'][i]['geometry']['coordinates'][0]) != float:
        beta.append(json_df['features'][i]['geometry']['coordinates'])

delta = []
for a in range(len(beta)):
    for i in range(len(beta[a])):
        delta.append(beta[a][i])

경도d = []
위도d = []
for i in range(len(delta)):
    경도d.append(delta[i][0])
    위도d.append(delta[i][1])

location_data_d = []
for i in range(len(경도d)):
    location_data_d.append([위도d[i], 경도d[i]])

경도 = []
위도 = []
for i in range(0, len(경도d), round(len(경도d)/30)):
    경도.append(경도d[i])
    위도.append(위도d[i])
df_좌표_2 = pd.DataFrame({'경도' : 경도, '위도' : 위도})

##
img_list = get_image(df_좌표_2)
sample_list = []
for i in range(len(img_list)):
    sample_list.append(transform(np.array(img_list[i])))

sample_list = np.array(sample_list)

model = load_model('traffic_image_predict_model')
pred = model.predict(sample_list)
center = location_data_d[round(len(df_좌표_2)/2)]

# def model_load():
map = folium.Map(location=[center[0], center[1]], zoom_start=13)
# 시작
risk = pred

risk_list = []
for i in range(len(risk)):
    risk_list.append(risk[i][0])
df_좌표_2['위험도'] = risk_list

# 전체 경로 평균 위험도(%)
risk_mean = round((df_좌표_2['위험도'].mean())*100, 2)

# 경로 선 그리기를 위한 작업
marker_경도 = [str(x) for x in df_좌표_2['경도']]
경로_경도d = [str(x) for x in 경도d]
marker_위도 = [str(x) for x in df_좌표_2['위도']]
경로_위도d = [str(x) for x in 위도d]

num_list = []
for a in range(len(marker_경도)):
    num_list.append('num' + f'{a}')

for a in range(len(marker_경도)):
    num_list[a] = [i for i in range(len(경로_경도d)) if (marker_경도[a] in 경로_경도d[i]) & (marker_위도[a] in 경로_위도d[i])]

num_new = []
for i in range(len(num_list)):
    num_new.append(num_list[i][0])
    
# 시작지/도착지 마커 추가
folium.Marker([df_좌표_2['위도'][0], df_좌표_2['경도'][0]], 
            icon = folium.Icon(color='black', icon='play'),
            popup=f'<pre>이 전체 경로의 평균 위험도는 {risk_mean} 입니다.</pre>',
            tooltip = '출발!').add_to(map)
folium.Marker([df_좌표_2['위도'][len(df_좌표_2)-1], df_좌표_2['경도'][len(df_좌표_2)-1]], 
            icon = folium.Icon(color='black', icon='flag'), 
            tooltip = '도착!').add_to(map)

# 경로 마커 추가
for i in range(1, len(df_좌표_2)-1):
    if df_좌표_2['위험도'][i] < 0.4:
        color = 'blue'
        tooltip = '안전'
        icon='ok-sign'
    elif (df_좌표_2['위험도'][i] > 0.9):
        color = 'red'
        tooltip = '위험'
        icon='exclamation-sign'
    else:
        color = 'orange'
        tooltip = '주의'
        icon='question-sign'
    folium.Marker([df_좌표_2['위도'][i], df_좌표_2['경도'][i]]
                , icon = folium.Icon(color=color, icon=icon), tooltip = tooltip
                ).add_to(map)

# 선 그리기
for i in range(len(df_좌표_2)-1):
    if (df_좌표_2['위험도'][i] + df_좌표_2['위험도'][i+1])/2 < 0.4:
        color='blue'
    elif (df_좌표_2['위험도'][i] + df_좌표_2['위험도'][i+1])/2 > 0.9:
        color='red'
    else:
        color='orange'
    for a in range(num_new[i], len(경도d)-1):
        folium.PolyLine([location_data_d[a], location_data_d[a+1]], color=color).add_to(map)


#지도 띄우기
folium_static(map)


# VGG16 : 13개의 컨볼루션 레이어와 3개의 완전 연결 레이어(Dense)를 포함하여 16개의 가중치 레이어로 구성
# 전이학습 (남이 만든 모델 가져다 쓰기)

from tensorflow.keras.applications import VGG16

model = VGG16() #모델(16개) 가중치 모두 가져올 수 있음 

model.summary()

print(model.weights)


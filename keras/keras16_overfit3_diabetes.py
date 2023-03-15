from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. 데이터 
datasets = load_diabetes()
x = datasets.data       #datasets 안의 data    
y = datasets['target']  #datasets 안에 있는 target 가져오겠다 / x,y 의미(구조) 동일
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=10))
model.add(Dense(5, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          verbose=1)
#훈련하면서 데이터가 저장되어있음.(loss, val_loss).. / model.fit의 결과치들을 반환함 #hist
#print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000020610E47A60>
'''
print(hist.history) #저장된 데이터들의 히스토리

{'loss': [585.8532104492188, 584.0467529296875, 582.2438354492188, 580.4405517578125, 578.6473388671875, 576.859619140625, 575.069091796875, 
573.2874755859375, 571.513671875, 569.7298583984375, 567.9724731445312, 566.2015380859375, 564.4489135742188, 562.6965942382812, 560.9532470703125, 
559.210693359375, 557.4775390625, 555.750732421875, 554.031005859375, 552.3063354492188, 550.5960693359375, 548.8748779296875, 547.1683959960938, 
545.4681396484375, 543.7672119140625, 542.074462890625, 540.3800659179688, 538.6973876953125, 
537.003662109375, 535.3264770507812], 

'val_loss': [578.2281494140625, 576.4202270507812, 574.6019897460938, 572.804443359375, 570.9872436523438, 569.192138671875, 567.4169921875, 
565.6229248046875, 563.8358154296875, 562.0667724609375, 560.2930908203125, 558.5145263671875, 556.751953125, 555.0128173828125, 553.262939453125,
551.5155029296875, 549.768310546875, 548.0259399414062, 546.2862548828125, 544.5974731445312, 542.8463134765625, 541.1209716796875, 539.41259765625,
537.7207641601562, 536.0327758789062, 534.32080078125, 532.6172485351562, 530.9113159179688, 529.2340087890625, 527.5482177734375]}
'''

'''
# 그림그리기(그래프)
import matplotlib.pyplot as plt
import matplotlib
#plt.plot(y)  #x : epochs 순서(1,2...n), y값 loss -> x처럼 숫자가 순서대로 흘러갈 경우에는 명시하지 않아도 됨
#plt.show()
matplotlib.rcParams['font.family'] ='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] =False  # '-1'폰트깨짐 해결
#(구글링)한글깨짐 해결-> 시스템폰트에서 한글폰트로  변경  

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.',c='red', label='로스' ) 
plt.plot(hist.history['val_loss'], marker='.',c='blue', label='발_로스' )  #label을 legend()에 넣어도 됨
plt.title('비만') # '보스턴' 한글로 쓰면 글자 깨짐 
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend() #범례 추가
plt.grid() #격자선(그리드) 설정
plt.show()
#통상적으로, val_loss가 loss보다 성능 떨어짐 = 성능이 낮은거 기준(val_loss)으로 잡아야함 (더 빡빡한 기준이어야 하므로)
#epochs 커질수록 loss값 줄어드는 그래프 보여줌-> epochs 조절하면서 loss값 조정!
'''
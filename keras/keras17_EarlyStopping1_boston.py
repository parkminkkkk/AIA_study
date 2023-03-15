#EarlyStopping 
#어느시점부터 overfit걸림 -> 최소의 loss = 최적의 weight 찾기위해서 EarlyStopping

from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#1. 데이터 
datasets = load_boston()
x = datasets.data       #datasets 안의 data    
y = datasets['target']  #datasets 안에 있는 target 가져오겠다 / x,y 의미(구조) 동일
print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=650, test_size=0.3
)

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13)) #sigmoid : 0~1사이로 한정
model.add(Dense(10, activation='relu'))
model.add(Dense(15))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=25, mode='min', 
              verbose=1,
              restore_best_weights=True
             )  

#EarlyStopping을 val_loss를 기준으로 할거야 (통상적으로 loss보다 val_loss가 낫다 )
#n번을 참겠다(최저나올때까지) -> 통상적으로, 조금 높게 잡는 편
#val_loss의 최저값을 찾겠다 ->'min'  / cf)r2값 찾을땐 'max'
#verbose=1 : 끊긴 지점 확인(어디서 끊겼는지)    #Epoch 00035: early stopping  => 35번째에서 끊김 = 30번째가 최소 val_loss다

#Q. 저장된 가중치가 최적의 가중치인가?  
#A. No(x) 끊긴 지점에서 가중치 저장됨 -> (왜냐하면, 35번쨰에서 끊겼다면, 30번째가 최소값인데, 35번째 weight값을 저장하므로!)
#restore_best_weights=True : 최저값에서 저장   (디폴트=false)
    
hist = model.fit(x_train, y_train, epochs=10000, batch_size=64,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#훈련하면서 데이터가 저장되어있음.(loss, val_loss).. / model.fit의 결과치들을 반환함 #hist
# print("============================================")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000020610E47A60>
# print("============================================")
# print(hist.history) #저장된 데이터들의 히스토리
# print("============================================")
# print(hist.history['loss']) 
# print("============================================")
# print(hist.history['val_loss']) 
# print("============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


'''
1. Epoch 00440: early stopping/ loss : 18.583890914916992/ r2스코어 :  0.6521450427599035
-patience=20, random_state=650874, Dense(32,16,4,2,1), activation'relu', mse, batch_size=100

-patience=100, random_state=650, Dense(32,16,4,2,1), activation'relu', mse, batch_size=32
'''



'''
# 그림그리기(그래프)
import matplotlib.pyplot as plt
import matplotlib
#plt.plot(y)  #x : epochs 순서(1,2...n), y값 loss -> x처럼 숫자가 순서대로 흘러갈 경우에는 명시하지 않아도 됨
#plt.show()
matplotlib.rcParams['font.family'] ='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] =False 
#(구글링)한글깨짐 해결-> 시스템폰트에서 한글폰트로  변경  

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.',c='red', label='로스' ) 
plt.plot(hist.history['val_loss'], marker='.',c='blue', label='발_로스' )  #label을 legend()에 넣어도 됨
plt.title('보스턴') # '보스턴' 한글로 쓰면 글자 깨짐 
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend() #범례 추가
plt.grid() #격자선(그리드) 설정
plt.show()
'''


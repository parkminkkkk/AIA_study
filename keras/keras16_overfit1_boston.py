#그림그리기(그래프), 한글폰트 -> 그래프를 보면서 과적합 / epochs조절하면서 loss값 조정

from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. 데이터 
datasets = load_boston()
x = datasets.data       #datasets 안의 data    
y = datasets['target']  #datasets 안에 있는 target 가져오겠다 / x,y 의미(구조) 동일
print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=13))
model.add(Dense(5, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
          validation_split=0.2,
          verbose=1)

#훈련하면서 데이터가 저장되어있음.(loss, val_loss).. / model.fit의 결과치들을 반환함 #hist
print("============================================")
print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000020610E47A60>
print("============================================")
print(hist.history) #저장된 데이터들의 히스토리
print("============================================")
print(hist.history['loss']) 
print("============================================")
print(hist.history['val_loss']) 
print("============================================")


# 그림그리기(그래프)
import matplotlib.pyplot as plt
import matplotlib
#plt.plot(y)  #x : epochs 순서(1,2...n), y값 loss -> x처럼 숫자가 순서대로 흘러갈 경우에는 명시하지 않아도 됨
#plt.show()
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
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
#통상적으로, val_loss가 loss보다 성능 떨어짐 = 성능이 낮은거 기준(val_loss)으로 잡아야함 (더 빡빡한 기준이어야 하므로)
#epochs 커질수록 loss값 줄어드는 그래프 보여줌-> epochs 조절하면서 loss값 조정!

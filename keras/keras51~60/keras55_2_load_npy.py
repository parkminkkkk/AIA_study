import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr= xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr= xy_train[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr= xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr= xy_train[0][1])

x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')

# print(x_train)
print(x_train.shape, x_test.shape) #(160, 100, 100, 1) (160, 100, 100, 1)
print(y_train.shape, y_test.shape) #(160,) (160,)



#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(100,100,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#3)fit
hist = model.fit(x_train,y_train, epochs=1000,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
                    steps_per_epoch=10,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
                    validation_data=[x_test, y_test],
                    validation_steps=24,  # val(test)데이터/batch = 120/5=24
                    )  

#history=(metrics)loss, val_loss, acc
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc) 
print("acc:",acc[-1])
print("val_acc:",val_acc[-1])
print("loss:",loss[-1])
print("val_loss:",val_loss[-1])


#[실습1]그림그리기 subplot(두개 그림을 하나로)
#[실습] 튜닝 acc 0.95이상


#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# plt.subplot(nrows,ncols,index)
# plt.subplot(총 행 개수, 총 열 개수, 그래프 번호)
plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(hist.history['loss'],marker='.', label='loss', c='red')
plt.plot(hist.history['val_loss'], marker='.', label='val_loss', c='blue')
plt.legend() #범례표시
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.grid() #격자표시

plt.subplot(1,2,2)
plt.title('Acc')
plt.plot(hist.history['acc'],  marker='.', label= 'acc', c='red')
plt.plot(hist.history['val_acc'], marker='.', label='val_acc', c='blue')
plt.legend()

plt.show()


'''
acc: 1.0
val_acc: 0.606249988079071
loss: 1.0756574297943189e-08
val_loss: 4.42763614654541

'''
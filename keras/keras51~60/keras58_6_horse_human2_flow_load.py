import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

save_path = 'd:/study_data/_save/keras58/'

x_train = np.load(save_path + 'keras58_6_x_train.npy')
x_test = np.load(save_path + 'keras58_6_x_test.npy')
y_train = np.load(save_path + 'keras58_6_y_train.npy')
y_test = np.load(save_path + 'keras58_6_y_test.npy')

#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(150,150,3), activation='relu'))
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
hist = model.fit(x_train,y_train, epochs=10,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
                    # steps_per_epoch=10,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
                    validation_data=[x_test, y_test],
                    batch_size = 32
                    # validation_steps=24,  # val(test)데이터/batch = 120/5=24
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

'''
acc: 0.7346329689025879
val_acc: 0.5
loss: 0.5073550343513489
val_loss: 0.6962357759475708
'''
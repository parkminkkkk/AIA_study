# 함수형 맹그러봐 


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score 



#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.


#2. 모델 
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3))

vgg16.trainable = False

# Define the input tensor
inputs = Input(shape=(32, 32, 3))
x = vgg16(inputs)
x = Flatten()(x)
# x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
outputs = Dense(100, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)


# model.trainable = True   ##vgg16만 가중치 동결(가져온 모델은 가중치 동결하고, 밑에 새로만든 dense는 가중치 형성해줌) 

# model.summary()
# print(len(model.weights))
# print(len(model.trainable_weights))

#3. 컴파일, 훈련 
# model.compile(loss = "mse", optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20, mode = 'min', verbose=1,)
rlr = ReduceLROnPlateau(monitor='val_loss', patience = 10, mode ='auto', verbose=1, factor=0.5)   #es, rlr의 patience는 따로 준다

model.fit(x_train, y_train, epochs =50, batch_size=128, verbose=1, validation_split=0.2,
            callbacks = [es, rlr])


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("loss:", results[0])
print("acc:", results[1])


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

###########################
#vgg16.trainable = False  



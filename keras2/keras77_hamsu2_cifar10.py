import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score 



#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.


#2. 모델 
base_model = VGG16(weights='imagenet', include_top=False, 
                   input_shape=(32,32,3)
                   )
# print(base_model.output)  #마지막 레이어 
# KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), 
#                                   dtype=tf.float32, name=None), name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")
x = base_model.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

model.summary()



#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20, mode = 'min', verbose=1,)
rlr = ReduceLROnPlateau(monitor='val_loss', patience = 10, mode ='auto', verbose=1, factor=0.5)   #es, rlr의 patience는 따로 준다

model.fit(x_train, y_train, epochs =50, batch_size=512, verbose=1, validation_split=0.2,
            callbacks = [es, rlr])


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("loss:", results[0])
print("acc:", results[1])


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

''''
Epoch 00020: early stopping
313/313 [==============================] - 3s 10ms/step - loss: nan - acc: 0.0100
loss: nan
acc: 0.009999999776482582
acc: 0.01
'''


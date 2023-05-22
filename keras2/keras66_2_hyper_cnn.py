#CNN으로 만들기

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,  Input, Dropout 
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28,1).astype('float32')/255.   #. : float형태로 형 변환 됨
x_test = x_test.reshape(-1, 28, 28,1).astype('float32')/255.     #(10000, 28, 28)
print(x_test.shape)



#2. 모델구성

def build_model(drop=0.3, optimizer='adam', activation='relu', 
                filters = 32, lr = 0.01):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(filters = 32, kernel_size= (2,2), padding='same', activation=activation, name = 'Conv1')(inputs)
    x = MaxPool2D()(x)
    x = Conv2D(filters,(2,2), activation=activation, name = 'Conv2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters,(2,2), activation=activation, name = 'Conv3')(x)
    x = Flatten()(x)
    x = Dense(24, activation=activation, name = 'dense1')(x)
    x = Dense(24, activation=activation, name = 'dense2')(x)
    outputs = Dense(10, activation='softmax', name = 'output')(x)

    model = Model(inputs = inputs, outputs=outputs)

    model.compile(optimizer='adam', metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model 

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    lr = [0.001, 0.005, 0.01]
    optimizers = [Adam(learning_rate=lr), RMSprop(learning_rate=lr), Adadelta(learning_rate=lr)]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'lr' : lr}


hyperparameters = create_hyperparameter()
print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #keras에서 sklearn사용 할 수 있게 rapping 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

keras_model = KerasClassifier(build_fn=build_model, verbose =1,) #, epochs = 3

es = EarlyStopping(monitor='val_loss', mode = 'min', patience=5, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint('./_save/MCP/keras66_cnn_best_model.h5', 
                      monitor='val_loss', mode='auto',
                      save_best_only=True, verbose=1)

# model = GridSearchCV(keras_model, hyperparameters, cv=3)  # build_model이 아니라 한번 랩핑한 keras_model을 넣어야함 
model = RandomizedSearchCV(keras_model, hyperparameters, cv=5, n_iter=3, verbose=1)


import time
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es, mcp])
end = time.time()


print('걸린시간 : ', end - start)
best_params = model.best_params_.copy()
best_params['optimizer'] = best_params['optimizer'].__class__.__name__
print('model.best_params_ :', best_params)
print('model.best_estimator : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score:', accuracy_score(y_test, y_predict))


'''
걸린시간: 298.5751254558563
model.best_params_: {'optimizer': 'adadelta', 'lr': 0.01, 'drop': 0.3, 'batch_size': 300, 'activation': 'selu'}
model.best_estimator_: <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A689A055E0>
model.best_score_: 0.8675333261489868
34/34 [==============================] - 0s 11ms/step - loss: 0.3572 - acc: 0.9052
model.score: 0.9052000045776367
acc_score: 0.9052
'''

'''
Epoch 00015: early stopping
걸린시간 :  383.1289610862732
model.best_params_ : {'optimizer': 'Adam', 'lr': 0.001, 'drop': 0.2, 'batch_size': 200, 'activation': 'relu'}
model.best_estimator :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F2CF8F6AF0>
model.best_score_ :  0.983216667175293
50/50 [==============================] - 0s 4ms/step - loss: 0.0461 - acc: 0.9864
model.score :  0.9864000082015991
acc_score: 0.9864
'''
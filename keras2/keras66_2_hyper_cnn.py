#CNN으로 만들기
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,  Input, Dropout 

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

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model 

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    learning_rates = [0.001, 0.01, 0.1]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop': dropouts,
            'activation': activations,
            'lr' : learning_rates}

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


print("걸린시간:", end - start)  
print("model.best_params_:", model.best_params_)    # 그리드 서치나 랜덤 서치와 같은 하이퍼파라미터 튜닝 과정에서 최적의 매개변수 조합 //  #탐색 과정에서 검증 데이터를 사용하여 최적의 매개변수를 찾은 후에 접근 가능
print("model.best_estimator_:", model.best_estimator_)  # 최적의 매개변수 조합으로 훈련된 모델=> 즉, best_params_에 해당하는 매개변수로 훈련된 모델 객체 //# best_estimator_는 최적의 모델을 얻을 수 있도록 하이퍼파라미터 튜닝 과정에서 사용
print("model.best_score_:", model.best_score_)      #train의 best_score
print("model.score:", model.score(x_test, y_test))  #test의 best_score

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
Epoch 00007: early stopping
걸린시간: 96.29829454421997
model.best_params_: {'optimizer': 'rmsprop', 'lr': 0.1, 'drop': 0.2, 'batch_size': 100, 'activation': 'selu'}
model.best_estimator_: <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002530A7EAA00>
model.best_score_: 0.9694333374500275
100/100 [==============================] - 0s 3ms/step - loss: 0.0627 - acc: 0.9820
model.score: 0.9819999933242798
acc_score: 0.982
'''
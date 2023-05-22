import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,  Input, Dropout 

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.   #. : float형태로 형 변환 됨
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255.   #(10000, 784)
# print(x_test.shape)

#2. 모델구성

def build_model(drop=0.3, optimizer='adam', activation='relu', 
                node1 = 64, node2 = 64, node3 = 64, lr = 0.001):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden4')(x)
    outputs = Dense(10, activation='softmax', name = 'output')(x)

    model = Model(inputs = inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy') #y원핫 안해줘도 됨 
    return model 

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop': dropouts,
            'activation': activations}

hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'dropout': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear']}


#TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <keras.engine.functional.Functional object at 0x0000026B0C708940> does not.
#keras, sklearn 만든 형태가 서로 공유가 가능할까?? =>> keras, sklearn이 제공하는 방식이 서로 다름 


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #keras에서 sklearn사용 할 수 있게 rapping 
keras_model = KerasClassifier(build_fn=build_model, verbose =1) #, epochs = 3

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model1 = build_model()
# model = GridSearchCV(keras_model, hyperparameters, cv=3)  # build_model이 아니라 한번 랩핑한 keras_model을 넣어야함 
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs = 3) #epochs = 3 keras_model 혹은 model.fit 두개 모두 epochs 가능 
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
걸린시간: 12.836474895477295
model.best_params_: {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 400, 'activation': 'relu'}
model.best_estimator_: <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002BD36C16910>
model.best_score_: 0.9597166478633881
25/25 [==============================] - 0s 3ms/step - loss: 0.0907 - acc: 0.9724
model.score: 0.9724000096321106
acc_score: 0.9735
'''




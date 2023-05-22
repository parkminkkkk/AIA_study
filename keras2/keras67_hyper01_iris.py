import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,  Input, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=337, shuffle=True
)

#2. 모델구성

def build_model(drop=0.3, optimizer='adam', activation='relu', 
                node1 = 512, node2 = 256, node3 = 128, node4 = 256, lr = 0.01):
    inputs = Input(shape=(4), name='input')
    x = Dense(node1, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name = 'hidden4')(x)
    outputs = Dense(1, activation='softmax', name = 'output')(x)

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
keras_model = KerasClassifier(build_fn=build_model, verbose =1,) #, epochs = 3

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = GridSearchCV(keras_model, hyperparameters, cv=3)  # build_model이 아니라 한번 랩핑한 keras_model을 넣어야함 

# Create the RandomizedSearchCV object with early stopping
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# mcp = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])
end = time.time()


print("+======================================+")
print("걸린시간:", end - start)  
print("model.best_params_:", model.best_params_)    # 그리드 서치나 랜덤 서치와 같은 하이퍼파라미터 튜닝 과정에서 최적의 매개변수 조합 //  #탐색 과정에서 검증 데이터를 사용하여 최적의 매개변수를 찾은 후에 접근 가능
print("model.best_estimator_:", model.best_estimator_)  # 최적의 매개변수 조합으로 훈련된 모델=> 즉, best_params_에 해당하는 매개변수로 훈련된 모델 객체 //# best_estimator_는 최적의 모델을 얻을 수 있도록 하이퍼파라미터 튜닝 과정에서 사용
print("model.best_score_:", model.best_score_)      #train의 best_score
print("model.score:", model.score(x_test, y_test))  #test의 best_score

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score:', accuracy_score(y_test, y_predict))


'''
Epoch 00003: early stopping
+======================================+
걸린시간: 4.629993200302124
model.best_params_: {'optimizer': 'adam', 'lr': 0.1, 'drop': 0.4, 'batch_size': 400, 'activation': 'relu'}
model.best_estimator_: <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016710A3BE80>
model.best_score_: 0.36666665971279144
1/1 [==============================] - 0s 22ms/step - loss: nan - acc: 0.2000
model.score: 0.20000000298023224
acc_score: 0.2
'''
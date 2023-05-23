from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import time


datalist = [load_iris(), load_breast_cancer()]

for i in datalist:
    x = datalist.data
    y = datalist.target

    print(f"Dataset: {datalist}")
    print(f"X shape: {x.shape}, Y shape: {y.shape}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=337, shuffle=True
    )


    def build_model(x, y, drop=0.3, optimizer='adam', activation='relu', 
                    node1=512, node2=256, node3=128, node4=256, lr=0.01):
        inputs = Input(shape=(x.shape[1],), name='input')
        x = Dense(node1, activation=activation, name='hidden1')(inputs)
        x = Dropout(drop)(x)
        x = Dense(node2, activation=activation, name='hidden2')(x)
        x = Dropout(drop)(x)
        x = Dense(node3, activation=activation, name='hidden3')(x)
        x = Dropout(drop)(x)
        x = Dense(node4, activation=activation, name='hidden4')(x)
        outputs = Dense(1, activation='softmax', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='sparse_categorical_crossentropy')
        return model

    def create_hyperparameter():
        batchs = [100, 200, 300, 400, 500]
        optimizers = ['adam', 'rmsprop', 'adadelta']
        dropouts = [0.2, 0.3, 0.4, 0.5]
        activations = ['relu', 'elu', 'selu', 'linear']
        learning_rates = [0.001, 0.01, 0.1]
        return {'batch_size': batchs,
                'optimizer': optimizers,
                'drop': dropouts,
                'activation': activations,
                'lr': learning_rates}

    hyperparameters = create_hyperparameter()

    keras_model = KerasClassifier(build_fn=build_model, verbose=1)

    model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    start = time.time()
    model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es], verbose=1)
    end = time.time()

    print("+======================================+")
    print("Elapsed Time:", end - start)
    print("Best Params:", model.best_params_)
    print("Best Estimator:", model.best_estimator_)
    print("Best Train Score:", model.best_score_)
    print("Test Score:", model.score(x_test, y_test))

    y_predict = model.predict(x_test)
    print('Accuracy Score:', accuracy_score(y_test, y_predict))




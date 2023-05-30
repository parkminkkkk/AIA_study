import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 데이터 로드 및 전처리
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 112.5
x_test = x_test.reshape(10000, 784).astype('float32') / 122.5

# 인코더 모델 정의
def get_encoder(hidden_units, activation):
    input_img = Input(shape=(784,))
    encoded = Dense(hidden_units, activation=activation)(input_img)
    return Model(input_img, encoded)

# 디코더 모델 정의
def get_decoder(hidden_units, activation):
    encoded_input = Input(shape=(hidden_units,))
    decoded = Dense(784, activation=activation)(encoded_input)
    return Model(encoded_input, decoded)

# 모델 조합하여 결과 확인
hidden_units_list = [1, 32, 64, 1024]
activation_list = ['relu', 'sigmoid', 'linear', 'tanh']

for hidden_units in hidden_units_list:
    for activation in activation_list:
        print(f"Hidden Units: {hidden_units}, Activation: {activation}")
        encoder = get_encoder(hidden_units, activation)
        decoder = get_decoder(hidden_units, activation)
        autoencoder = Model(encoder.input, decoder(encoder.output))
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

        decoded_imgs = autoencoder.predict(x_test)

        import matplotlib.pyplot as plt

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

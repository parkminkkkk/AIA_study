import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

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
hidden_units_list = [1,32,64,1024]   #1,32,64,1024
activation_list = ['relu', 'sigmoid', 'linear', 'tanh'] #'relu', 'sigmoid', 'linear', 'tanh'

n = 10  # 각 경우의 수당 그림 개수
num_cases = len(hidden_units_list) * len(activation_list)
fig, axes = plt.subplots(num_cases, 2 * n, figsize=(20, 4 * num_cases))

row = 0

for hidden_units in hidden_units_list:
    for activation in activation_list:
        encoder = get_encoder(hidden_units, activation)
        decoder = get_decoder(hidden_units, activation)
        autoencoder = Model(encoder.input, decoder(encoder.output))
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(x_train, x_train, epochs=, batch_size=128, validation_split=0.2)

        decoded_imgs = autoencoder.predict(x_test)

        for i in range(n):
            ax = axes[row, i]
            ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
            ax.axis('off')

            ax = axes[row, i + n]
            ax.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
            ax.axis('off')

        axes[row, 0].text(-80, 12, f"Hidden Units: {hidden_units}", fontsize=10, ha='center', va='center')
        axes[row, 0].text(-80, -15, f"Activation: {activation}", fontsize=10, ha='center', va='center')

        row += 1

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# 1. Load and prepare MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Add simple noise
x_train_noisy = x_train + 0.3 * np.random.randn(*x_train.shape)
x_test_noisy = x_test + 0.3 * np.random.randn(*x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 3. Very simple model: one conv + one upsample
model = Sequential([
    Conv2D(8, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    Conv2D(8, (3,3), activation='relu', padding='same'),
    UpSampling2D((1,1)),
    Conv2D(1, (3,3), activation='sigmoid', padding='same')
])

# 4. Compile and train
model.compile(optimizer=Adam(), loss='mse')
model.fit(x_train_noisy, x_train, epochs=3, batch_size=128)

# 5. Predict and show
output = model.predict(x_test_noisy)

# 6. Show noisy vs denoised
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title("Noisy")

    plt.subplot(2,5,i+6)
    plt.imshow(output[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title("Cleaned")

plt.tight_layout()
plt.show()
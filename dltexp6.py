import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

preds = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].numpy())
    plt.title(f"Predicted: {np.argmax(preds[i])}, True: {np.argmax(y_test[i])}")
    plt.axis('off')
    plt.show()

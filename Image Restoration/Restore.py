!pip install tensorflow keras matplotlib opencv-python-headless

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate


dataset_directory = '/content/drive/My Drive/degraded'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64)))
    return images

images = load_images_from_folder(dataset_directory)

preprocessed_images = np.array(images).astype('float32') / 255.
preprocessed_images = np.reshape(preprocessed_images, (len(preprocessed_images), 64, 64, 1))

split = int(len(preprocessed_images) * 0.8)  
x_train, x_val = preprocessed_images[:split], preprocessed_images[split:]


input_img = Input(shape=(64, 64, 1))


x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x1 = MaxPooling2D((2, 2), padding='same')(x1)
x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPooling2D((2, 2), padding='same')(x2)


bottleneck = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)


x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bottleneck)
x2 = UpSampling2D((2, 2))(x2)
x2 = concatenate([x2, x1])
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
x1 = UpSampling2D((2, 2))(x1)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x1)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


autoencoder.fit(x_train, x_train,
                epochs=50,  
                batch_size=8,
                shuffle=True,
                validation_data=(x_val, x_val))


val_images_restored = autoencoder.predict(x_val)
mse = mean_squared_error(x_val.flatten(), val_images_restored.flatten())
print(f"Mean Squared Error on validation set: {mse}")

plt.figure(figsize=(20, 4))
for i in range(10):  
  
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_val[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(val_images_restored[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


restored_images = autoencoder.predict(x_val[:10])



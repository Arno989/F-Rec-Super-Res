import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import urllib
import platform
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt


"""   -----   Checkpoint callback   -----   """
checkpoint_path = f"./ML/Checkpoints"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, f"cp.ckpt"), verbose=1, save_weights_only=True, save_freq=5) # -{epoch:02d}



"""   -----   Variables   -----   """
l2_alpha = 10e-10  # L2 regression



"""   -----   Model   -----   """
# conv == size of convolution window in px
def define_model(conv=8):
    input_img = tf.keras.layers.Input(shape=(1024, 1024, 3))

    l1 = tf.keras.layers.Conv2D(256, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(input_img)
    l2 = tf.keras.layers.Conv2D(256, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l1)
    l3 = tf.keras.layers.MaxPool2D(padding="same")(l2)

    l4 = tf.keras.layers.Conv2D(512, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l3)
    l5 = tf.keras.layers.Conv2D(512, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l4)
    l6 = tf.keras.layers.MaxPool2D(padding="same")(l5)

    l7 = tf.keras.layers.Conv2D(1024, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l6)

    l8 = tf.keras.layers.UpSampling2D()(l7)
    l9 = tf.keras.layers.Conv2D(512, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l8)
    l10 = tf.keras.layers.Conv2D(512, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l9)

    l11 = tf.keras.layers.add([l10, l5])

    l12 = tf.keras.layers.UpSampling2D()(l11)
    l13 = tf.keras.layers.Conv2D(256, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l12)
    l14 = tf.keras.layers.Conv2D(256, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l13)

    l15 = tf.keras.layers.add([l14, l2])

    decoded_image = tf.keras.layers.Conv2D(3, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha),)(l15)

    return tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)



"""   -----   Save empty weights   -----   """
model = define_model()
model.save_weights(os.path.join(checkpoint_path, "cp-00.ckpt"))
model.compile(optimizer="adam", loss="mean_squared_error")



"""   -----   Load data   -----   """
lo_res_images = []
hi_res_images = []

t, t, hi_files = next(os.walk("./Data/High res"))
t, t, lo_files = next(os.walk("./Data/Low res"))

for img in hi_files:
    hi_res_images.append(cv2.imread(f"./Data/High res/{img}", cv2.IMREAD_UNCHANGED))
    
for img in lo_files:
    lo_res_images.append(cv2.imread(f"./Data/Low res/{img}", cv2.IMREAD_UNCHANGED))



"""   -----   Fit, save & test model   -----   """
model.fit(np.asarray(lo_res_images[:5]), np.asarray(hi_res_images[:5]), epochs=1, batch_size=2, shuffle=True, validation_split=0.15)   # Test mdel
model.fit(np.asarray(lo_res_images), np.asarray(hi_res_images), epochs=10, batch_size=5, shuffle=True, validation_split=0.3, callbacks=[cp_callback])   # Train mdel

model.save("my_model")
sr1 = np.clip(model.predict(lo_res_images), 0.0, 1.0)



"""   -----   Show results   -----   """
plt.figure(figsize=(256, 256))

plt.subplot(10, 10, 1)
plt.imshow(lo_res_images[image_index])

plt.subplot(10, 10, 2)
plt.imshow(hi_res_images[image_index])

plt.subplot(10, 10, 3)
plt.imshow(sr1[image_index])

plt.savefig("./")
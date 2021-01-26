import tensorflow as tf
import numpy as np











conv1 = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
print()
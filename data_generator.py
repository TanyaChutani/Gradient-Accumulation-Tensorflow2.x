from numpy.core.fromnumeric import shape
import os
import numpy as np
import tensorflow as tf

class DataGenerator:
  def __init__(
      self,
      dataset_name,
      batch_size,
      resize_dim,
      mode,
      shuffle=True,
      n_channels=3,
  ):
    self.mode = mode
    self.dataset_name = dataset_name
    self.batch_size = batch_size
    self.resize_dim = resize_dim
    self.shuffle = shuffle
    self.n_channels = n_channels

  def __len__(self):
      return int(np.ceil((self.images).shape[0] / self.batch_size))

  def __call__(self):
    data_name = self.dataset_name 
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.data_name.load_data()
    self.images, self.labels = (x_train,y_train) if self.mode == "train" else (x_test,y_test)

    for i in self.index:
      print(self.index)
      x, y = self.load(self.preprocess(self.images[i]), self.labels[i])
      yield x, y

  def on_epoch_end(self):
      self.index = np.arange((self.images).shape[0])
      if self.shuffle == True:
          np.random.shuffle(self.index)

  def preprocess(self, image):
      image = tf.image.resize(
          image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
      )
      image = image / 255.0
      image = tf.cast(image, tf.float32)
      return image

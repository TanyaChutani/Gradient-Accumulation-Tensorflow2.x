import tensorflow as tf

def model(num_classes):  
  model_input = tf.keras.layers.Input(shape=(32,32,3),dtype=tf.float32)
  model = tf.keras.applications.ResNet50(include_top=False,
                                            pooling='avg',
                                         input_tensor=model_input)
  model.trainable = False
  predictions = tf.keras.layers.Dense(num_classes,activation="softmax")(model.output)
  return tf.keras.Model(inputs=model.input, outputs=predictions)

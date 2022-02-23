import tensorflow as tf

def data_generator(features,labels,batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((tf.cast((features/255),tf.float32),labels))
  dataset = dataset.shuffle(buffer_size=len(labels)+1)
  dataset = dataset.batch(batch_size=batch_size,
                          drop_remainder=True)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

import tensorflow as tf

class Classification(tf.keras.models.Model):
  def __init__(self, num_classes=10.0, **kwargs):
    super(Classification, self).__init__()
    self.num_classes = num_classes
    self._gradients = []
    self._accum_steps = tf.Variable(tf.constant(0, dtype=tf.int64),
                                            trainable=False,
                                            synchronization=tf.VariableSynchronization.ON_READ,
                                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                            )


  def compile(self, optimizer, metric, loss_fn):
      super(Classification, self).compile()
      self.metric = metric
      self.optimizer = optimizer
      self.loss_fn = loss_fn

  def build(self, input_shape) -> None:
      self.resnet = tf.keras.applications.ResNet50(
          include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling='max',
          classes=self.num_classes,
      )
      self.output_layer = tf.keras.layers.Dense(self.num_classes)
  
  def call(self, input_tensor):
    x = self.resnet(input_tensor)
    x = self.output_layer(x)
    return x

  def train_step(self, data):
    images, labels = data

    with tf.GradientTape() as tape:

        model_output = self(images, training=True)
        loss = tf.reduce_mean(self.loss_fn(labels, model_output))
        acc = self.metric(labels, tf.nn.softmax(model_output))
    
    gradients = tape.gradient(loss, self.trainable_variables)


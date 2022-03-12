import tensorflow as tf

class Classification(tf.keras.models.Model):
  def __init__(self, num_classes=10.0, **kwargs):
    super(Classification, self).__init__()
    self.num_classes = num_classes
    
  def compile(self, optimizer, metric, loss_fn):
      super(Classification, self).compile()
      self.metric = metric
      self.optimizer = optimizer
      self.loss_fn = loss_fn

  def build(self, input_shape):
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

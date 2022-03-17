import tensorflow as tf
import tensorflow as tf


class Classification(tf.keras.models.Model):
    def __init__(self, num_gradients, num_classes=10, width=32, height=32):
        super(Classification, self).__init__()
        self.width = tf.constant(width, dtype=tf.int32)
        self.height = tf.constant(height, dtype=tf.int32)
        self.num_classes = tf.constant(num_classes, dtype=tf.int32)
        self.num_gradients = tf.constant(num_gradients, dtype=tf.int32)
        self.accum_steps = tf.Variable(
            tf.constant(0, dtype=tf.int32),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.gradient_accumulation = [
            tf.Variable(
                tf.zeros_like(v, dtype=tf.float32),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
            )
            if v is not None
            else v
            for v in self.trainable_variables
        ]
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=((self.height, self.width, 3)),
            pooling="max",
            classes=self.num_classes,
        )
        self.output_layer = tf.keras.layers.Dense(self.num_classes)

    def compile(self, optimizer, metric, loss):
        super(Classification, self).compile()
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer

    def call(self, input_tensor):
        x = self.resnet(input_tensor)
        x = self.output_layer(x)
        return x

    def reset(self):
        if not self.gradient_accumulation:
            return

        self.accum_steps.assign_add(0)

        for gradient in self.gradient_accumulation:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)

    def accumulate_gradients(self):
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )
        self.reset()

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:

            model_output = self(images, training=True)
            loss = tf.reduce_mean(self.loss(labels, model_output))
            acc = self.metric(labels, model_output)

        gradients = tape.gradient(loss, self.trainable_variables)

        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        tf.cond(
            tf.equal(self.accum_steps, self.num_gradients),
            self.accumulate_gradients,
            lambda: None,
        )
        return {"loss": loss}
